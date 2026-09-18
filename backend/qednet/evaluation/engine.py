"""F7 — Automated Benchmarking Engine.

One central engine for classical AND quantum models:
dataset -> validation -> nested cross-validation (outer k, inner tuning) ->
multi-seed -> metrics -> statistics -> calibration -> runtime/resource
accounting. The SAME evaluation framework is used for every model; results
are stored in machine-readable JSON with full provenance.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import ExperimentConfig
from ..data.validation import (LeakageGuard, check_train_test_contamination,
                               to_binary_target, validate_frame)
from ..models.registry import CLASSICAL_MODELS, QUANTUM_MODELS, build_model
from ..preprocess.pipeline import PreprocessPipeline
from .calibration import calibration_report, fit_calibrator
from .metrics import (compute_metrics, pr_curve_data, reliability_curve_data,
                      roc_curve_data)
from .statistics import (bootstrap_ci, corrected_cv_ttest, decision_curve_data,
                         delong_test)

logger = logging.getLogger("qednet.engine")

# Small, controlled, logged hyperparameter grids (inner CV tuning budgets).
INNER_GRIDS: Dict[str, Dict[str, list]] = {
    "logistic_regression": {"C": [0.1, 1.0, 10.0]},
    "rbf_svm": {"C": [0.5, 1.0, 4.0]},
    "random_forest": {"n_estimators": [200, 400]},
    "xgboost": {"max_depth": [3, 5]},
    "matched_mlp": {},
    "qsvm": {},
    "vqc": {},
    "reupload": {},
    "hybrid_qnn": {},
}


@dataclass
class FoldRecord:
    model: str
    seed: int
    fold: int
    metrics: Dict[str, float]
    y_test: List[int]
    p_test: List[float]
    p_calibrated: List[float]
    resources: Optional[Dict[str, Any]]
    model_meta: Dict[str, Any]
    leakage_audit: Dict[str, Any]


def default_suite(cfg: ExperimentConfig) -> List[str]:
    if cfg.models:
        return list(cfg.models)
    return CLASSICAL_MODELS + QUANTUM_MODELS


class BenchmarkEngine:
    """Nested-CV multi-seed benchmark engine (model-agnostic)."""

    def __init__(self, cfg: ExperimentConfig, tuning_enabled: bool = True,
                 progress_cb=None):
        self.cfg = cfg
        self.tuning_enabled = tuning_enabled
        self.progress_cb = progress_cb
        self.run_started = None
        self.progress = {"stage": "benchmarking", "seed": None, "fold": None,
                         "model": None, "completed_fits": 0}

    # ------------------------------------------------------------------ run
    def run(self, df: pd.DataFrame, target: str,
            dataset_name: str) -> Dict[str, Any]:
        """Run the full benchmark; returns the machine-readable result."""
        self.run_started = time.time()
        validation = validate_frame(df, target, dataset_name)
        if not validation.passed:
            raise ValueError(
                f"Dataset validation failed: {validation.errors}")

        df = df.drop_duplicates()
        y_full = to_binary_target(df, target).to_numpy()
        feature_cols = [c for c in df.columns if c != target]
        X_full = df[feature_cols].to_numpy(dtype=np.float64)

        suite = default_suite(self.cfg)
        matched_params = self._quantum_param_budget()

        records: List[FoldRecord] = []
        curves: Dict[str, Dict[str, Any]] = {}
        for seed in self.cfg.training.seeds:
            outer = StratifiedKFold(
                n_splits=self.cfg.training.outer_folds, shuffle=True,
                random_state=seed)
            for fold_i, (tr, te) in enumerate(outer.split(X_full, y_full)):
                records += self._run_fold(
                    suite, X_full, y_full, tr, te, seed, fold_i,
                    matched_params, feature_cols)

        result = self._aggregate(records, curves, validation, dataset_name,
                                 suite, feature_cols)
        return result

    # ------------------------------------------------------------- one fold
    def _run_fold(self, suite, X_full, y_full, tr, te, seed, fold_i,
                  matched_params, feature_cols) -> List[FoldRecord]:
        X_tr, X_te = X_full[tr], X_full[te]
        y_tr, y_te = y_full[tr], y_full[te]
        out: List[FoldRecord] = []

        contamination = check_train_test_contamination(X_tr, X_te)
        if contamination["contaminated"]:
            raise RuntimeError(
                f"Train/test contamination in seed {seed} fold {fold_i}: "
                f"{contamination['n_shared_rows']} shared rows — "
                "EXPERIMENT INVALID")

        guard = LeakageGuard()
        pipe = PreprocessPipeline(
            compression_method=self.cfg.compression.method,
            target_dim=self.cfg.compression.target_dim,
            guard=guard)
        guard.allow_fit()
        Z_tr = pipe.fit_transform(X_tr, y_tr, feature_names=feature_cols)
        guard.forbid_fit(f"test-fold s{seed} f{fold_i}")
        Z_te = pipe.transform(X_te)

        for model_key in suite:
            kwargs = self._model_kwargs(model_key, matched_params)
            model = build_model(model_key, random_state=seed, **kwargs)
            if self.tuning_enabled and INNER_GRIDS.get(model_key):
                best = self._inner_tune(model_key, kwargs, Z_tr, y_tr, seed)
                kwargs = {**kwargs, **best}
                model = build_model(model_key, random_state=seed, **kwargs)
            guard.allow_fit()
            model.fit(Z_tr, y_tr)
            guard.forbid_fit(f"eval s{seed} f{fold_i}")
            p_te = model.predict_proba(Z_te)[:, 1]
            p_tr = model.predict_proba(Z_tr)[:, 1]

            # calibration fitted on TRAIN predictions only
            calibrator = fit_calibrator(self.cfg.evaluation.calibration,
                                        p_tr, y_tr)
            p_cal = calibrator.transform(p_te) if calibrator else p_te

            metrics = compute_metrics(y_te, p_te)
            metrics_cal = compute_metrics(y_te, p_cal)
            audit = guard.audit()

            rec = FoldRecord(
                model=model_key, seed=seed, fold=fold_i,
                metrics=metrics,
                y_test=[int(v) for v in y_te],
                p_test=[float(v) for v in p_te],
                p_calibrated=[float(v) for v in p_cal],
                resources=model.quantum_resources(),
                model_meta=model.metadata(),
                leakage_audit={
                    "fitted_parts": audit["fitted_parts"],
                    "contamination_detected": audit["contamination_detected"],
                    "train_test_shared_rows": contamination["n_shared_rows"],
                })
            rec.metrics["brier_calibrated"] = metrics_cal["brier"]
            rec.metrics["ece_calibrated"] = metrics_cal["ece"]
            out.append(rec)
            logger.info("seed=%d fold=%d model=%-20s auroc=%.4f",
                        seed, fold_i, model_key, metrics["auroc"])
            if self.progress_cb is not None:
                self.progress.update({
                    "seed": seed, "fold": fold_i, "model": model_key,
                    "completed_fits": self.progress["completed_fits"] + 1,
                    "auroc": round(float(metrics["auroc"]), 4),
                })
                try:
                    self.progress_cb(dict(self.progress))
                except Exception:  # noqa: BLE001 - progress is best-effort
                    pass
        return out

    # ------------------------------------------------------------ inner CV
    def _inner_tune(self, model_key: str, base_kwargs: Dict[str, Any],
                    Z_tr: np.ndarray, y_tr: np.ndarray,
                    seed: int) -> Dict[str, Any]:
        """Inner k-fold tuning inside the training fold (leakage-safe)."""
        grid = INNER_GRIDS[model_key]
        keys = list(grid)
        combos = [dict(zip(keys, v)) for v in _product(grid[k] for k in keys)]
        inner = StratifiedKFold(n_splits=self.cfg.training.inner_folds,
                                shuffle=True, random_state=seed)
        best_combo, best_score = None, -np.inf
        for combo in combos:
            aucs = []
            for itr, ite in inner.split(Z_tr, y_tr):
                m = build_model(model_key, random_state=seed,
                                **{**base_kwargs, **combo})
                m.fit(Z_tr[itr], y_tr[itr])
                p = m.predict_proba(Z_tr[ite])[:, 1]
                if len(np.unique(y_tr[ite])) == 2:
                    aucs.append(float(
                        __import__("sklearn.metrics", fromlist=["x"])
                        .roc_auc_score(y_tr[ite], p)))
            score = float(np.mean(aucs)) if aucs else -np.inf
            if score > best_score:
                best_score, best_combo = score, combo
        logger.info("inner tuning %s -> %s (inner auroc=%.4f)",
                    model_key, best_combo, best_score)
        return best_combo or {}

    # ------------------------------------------------------------ aggregate
    def _aggregate(self, records: List[FoldRecord], curves,
                   validation, dataset_name, suite, feature_cols
                   ) -> Dict[str, Any]:
        by_model: Dict[str, List[FoldRecord]] = {}
        for r in records:
            by_model.setdefault(r.model, []).append(r)

        models_out: Dict[str, Any] = {}
        for model_key, recs in by_model.items():
            pooled_y = np.concatenate([r.y_test for r in recs])
            pooled_p = np.concatenate([r.p_test for r in recs])
            pooled_pc = np.concatenate([r.p_calibrated for r in recs])

            summary = {}
            for metric in ("auroc", "auprc", "sensitivity", "specificity",
                           "sens_at_spec_95", "f1", "mcc", "brier",
                           "ece", "brier_calibrated", "ece_calibrated"):
                vals = np.array([r.metrics[metric] for r in recs], dtype=float)
                vals = vals[~np.isnan(vals)]
                summary[metric] = {
                    "mean": float(np.mean(vals)) if len(vals) else None,
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n_obs": int(len(vals)),
                }

            # bootstrap CI on pooled out-of-fold predictions
            bs = bootstrap_ci(pooled_y, pooled_p, "auroc",
                              n_resamples=self.cfg.evaluation.bootstrap_samples,
                              seed=self.cfg.training.seeds[0]) \
                if self.cfg.evaluation.bootstrap_enabled else None

            fpr, tpr = roc_curve_data(pooled_y, pooled_p)
            prec, rec_ = pr_curve_data(pooled_y, pooled_p)
            cal_centers, cal_accs, _ = reliability_curve_data(pooled_y, pooled_pc)
            dca = decision_curve_data(pooled_y, pooled_p) \
                if self.cfg.evaluation.decision_curve else None

            models_out[model_key] = {
                "model": model_key,
                "family": recs[0].model_meta.get("family", "unknown"),
                "summary": summary,
                "bootstrap_auroc": bs,
                "curve_roc": {"fpr": [round(float(v), 5) for v in fpr],
                              "tpr": [round(float(v), 5) for v in tpr]},
                "curve_pr": {"precision": [round(float(v), 5) for v in prec],
                             "recall": [round(float(v), 5) for v in rec_]},
                "curve_calibration": {"centers": cal_centers,
                                      "accuracy": cal_accs},
                "decision_curve": dca,
                "resources": recs[-1].resources,
                "model_meta": recs[-1].model_meta,
                "param_count": recs[-1].model_meta.get("n_parameters"),
                "train_time_s": {
                    "mean": float(np.mean(
                        [r.model_meta["train_time_s"] for r in recs])),
                    "total": float(np.sum(
                        [r.model_meta["train_time_s"] for r in recs]))},
                "leakage_audit": {
                    "fitted_parts": recs[0].leakage_audit["fitted_parts"],
                    "contamination_detected": any(
                        r.leakage_audit["contamination_detected"]
                        or r.leakage_audit["train_test_shared_rows"]
                        for r in recs),
                },
                "folds": [{
                    "seed": r.seed, "fold": r.fold,
                    "metrics": {k: (None if v != v else round(float(v), 6)
                                    if isinstance(v, float) else v)
                                for k, v in r.metrics.items()},
                } for r in recs],
                "pooled_predictions": {
                    "y": [int(v) for v in pooled_y],
                    "p": [round(float(v), 6) for v in pooled_p],
                    "p_calibrated": [round(float(v), 6) for v in pooled_pc],
                },
            }

        # pairwise DeLong tests: quantum vs best classical
        stats_out = self._pairwise_statistics(models_out, suite)

        # records span models × seeds × outer folds; each seed's test folds
        # partition the dataset exactly once, so dividing by (models × seeds)
        # recovers the true dataset size.
        n_models_in_records = len({r.model for r in records})
        runtime_s = time.time() - self.run_started
        return {
            "dataset": dataset_name,
            "n_samples": int(sum(len(r.y_test) for r in records) /
                             (n_models_in_records *
                              len(self.cfg.training.seeds))) if records else 0,
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "validation": validation.to_dict(),
            "protocol": {
                "outer_folds": self.cfg.training.outer_folds,
                "inner_folds": self.cfg.training.inner_folds,
                "seeds": self.cfg.training.seeds,
                "tuning": "inner-CV grid search (small logged grids)"
                if self.tuning_enabled else "disabled",
                "inner_grids": {k: v for k, v in INNER_GRIDS.items() if v},
            },
            "models": models_out,
            "statistics": stats_out,
            "runtime_s": round(runtime_s, 2),
        }

    def _pairwise_statistics(self, models_out, suite) -> Dict[str, Any]:
        classical_best = self._best(models_out, CLASSICAL_MODELS, "auroc")
        stats: Dict[str, Any] = {}
        q_keys = [k for k in suite if k in QUANTUM_MODELS]
        for qk in q_keys:
            if qk not in models_out:
                continue
            qm = models_out[qk]
            entry: Dict[str, Any] = {}
            if classical_best and self.cfg.evaluation.delong_test:
                y = np.array(qm["pooled_predictions"]["y"])
                p_q = np.array(qm["pooled_predictions"]["p"])
                p_c = np.array(
                    models_out[classical_best]["pooled_predictions"]["p"])
                # pooled predictions are aligned across models (same folds)
                entry["delong_vs_best_classical"] = delong_test(y, p_q, p_c)
                entry["best_classical_model"] = classical_best
            stats[qk] = entry

        # corrected CV comparison: each quantum vs best classical (per-fold)
        if classical_best and classical_best in models_out:
            c_recs = models_out[classical_best]["folds"]
            for qk in q_keys:
                if qk not in models_out:
                    continue
                q_recs = models_out[qk]["folds"]
                a = [f["metrics"]["auroc"] for f in q_recs
                     if f["metrics"]["auroc"] is not None]
                b = [f["metrics"]["auroc"] for f in c_recs
                     if f["metrics"]["auroc"] is not None]
                if len(a) == len(b) and len(a) >= 2:
                    n_test = len(models_out[qk]
                                 ["pooled_predictions"]["y"]) // max(1, len(a))
                    n_train = models_out[qk]["model_meta"]["n_train_samples"]
                    stats[qk]["corrected_cv_vs_best_classical"] = \
                        corrected_cv_ttest(a, b, n_train, n_test,
                                           self.cfg.training.outer_folds)
        return stats

    def _best(self, models_out, keys, metric):
        best, best_val = None, -np.inf
        for k in keys:
            if k in models_out:
                v = models_out[k]["summary"][metric]["mean"]
                if v is not None and v > best_val:
                    best, best_val = k, float(v)
        return best

    def _quantum_param_budget(self) -> int:
        m = self.cfg.model
        return int(m.layers * m.n_qubits + 1)

    def _model_kwargs(self, model_key: str, matched_params: int
                      ) -> Dict[str, Any]:
        m = self.cfg.model
        kwargs: Dict[str, Any] = {}
        if model_key == "matched_mlp":
            kwargs["match_params"] = matched_params
        if model_key in QUANTUM_MODELS:
            kwargs.update({
                "n_qubits": m.n_qubits, "layers": m.layers,
                "encoding": m.encoding if model_key != "qsvm"
                else ("zz" if m.encoding == "zz" else "angle"),
                "backend": m.backend, "entanglement": m.entanglement,
                "shots": self.cfg.training.shots,
            })
        return kwargs


def _product(iterables):
    import itertools
    return list(itertools.product(*iterables))
