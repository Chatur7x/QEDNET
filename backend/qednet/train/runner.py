"""Experiment runner: the complete F1 -> F12 end-to-end flow.

    Dataset -> F1 Ingestion -> F2 Validation + Leakage Guard ->
    F3 Preprocessing + Compression -> F4 Encoding -> F5/F6 Models ->
    F7 Benchmarking -> F8 Explainability -> F9 Certificate ->
    F10 Honesty Meter -> F11 Uncertainty -> F12 Cascade -> Report

Produces machine-readable JSON results, a Quantum Advantage Certificate,
research reports, and serialised deployment artifacts (for the prediction
view) with full provenance.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from ..certificate.ablation import entanglement_ablation
from ..certificate.certificate import (build_certificate,
                                       certificate_to_markdown)
from ..certificate.dequant import (nystrom_approximation,
                                   random_fourier_approximation)
from ..cascade.cascade import cascade_evaluate
from ..config import ExperimentConfig, config_to_dict
from ..data.datasets import load_dataset, register_builtins
from ..data.ingestion import DatasetRegistry, fingerprint_frame
from ..data.validation import to_binary_target, validate_frame
from ..evaluation.engine import BenchmarkEngine
from ..evaluation.engine import default_suite as _default_suite
from ..explain.explainer import (permutation_global, quantum_diagnostics,
                                 sensitivity_analysis, shap_global_local)
from ..honesty.meter import bottleneck_honesty_meter
from ..models.registry import CLASSICAL_MODELS, QUANTUM_MODELS, build_model
from ..models.training import quantum_input_transform
from ..preprocess.pipeline import PreprocessPipeline
from ..tracking.store import (ExperimentStore, reproducibility_snapshot)
from ..uncertainty.abstention import (AbstentionRule, ConformalClassifier,
                                      coverage_stats)

logger = logging.getLogger("qednet.runner")


def run_experiment(cfg: ExperimentConfig, store: ExperimentStore,
                   registry: DatasetRegistry,
                   explain_samples: int = 8,
                   run_explainability: bool = True,
                   log_level: int = logging.INFO) -> Dict[str, Any]:
    """Execute one full experiment and persist all artifacts."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t_start = time.time()

    exp_id = cfg.experiment_id
    exp_dir = store.experiment_dir(exp_id)

    # ---- F1: ingestion -----------------------------------------------------
    df = load_dataset(cfg.dataset.name, registry)
    target = registry.get(cfg.dataset.name)["target"]
    fingerprint = fingerprint_frame(df)

    # ---- F2: validation ------------------------------------------------------
    validation = validate_frame(df, target, cfg.dataset.name)
    if not validation.passed:
        raise ValueError(f"Validation failed: {validation.errors}")
    df = df.drop_duplicates()
    y = to_binary_target(df, target).to_numpy()
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    logger.info("[F1/F2] %s: %d samples x %d features (fingerprint %s)",
                cfg.dataset.name, X.shape[0], X.shape[1], fingerprint)

    # ---- F7: benchmark engine (F2-F6 inside) ---------------------------------
    def _progress(state: dict) -> None:
        store.save_json(exp_id, "progress", {
            **state, "updated_unix": int(time.time()),
            "seeds_total": len(cfg.training.seeds),
            "fits_per_seed": cfg.training.outer_folds * len(
                _default_suite(cfg))})

    engine = BenchmarkEngine(cfg, progress_cb=_progress)
    result = engine.run(df, target, cfg.dataset.name)
    result["dataset_fingerprint"] = fingerprint

    # reproducibility + config persistence
    repro = reproducibility_snapshot(cfg.config_hash())
    result["reproducibility"] = repro
    result["experiment_id"] = exp_id
    result["config"] = config_to_dict(cfg)
    result["research_status"] = (
        "RESEARCH AND EDUCATIONAL USE ONLY — not a medical device, not "
        "clinically validated, not for diagnosis.")
    store.save_result(exp_id, result)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg)))

    _progress({"stage": "refit_deployment_models", "completed_fits": None})

    # ---- deployment artifacts: refit on full data (for predict/explain) -----
    pipe_full = PreprocessPipeline(
        compression_method=cfg.compression.method,
        target_dim=cfg.compression.target_dim)
    Z_full = pipe_full.fit_transform(X, y, feature_names=feature_cols)
    matched_params = int(cfg.model.layers * cfg.model.n_qubits + 1)

    suite = list(result["models"].keys())
    fitted: Dict[str, Any] = {}
    for model_key in suite:
        kwargs = engine._model_kwargs(model_key, matched_params)
        m = build_model(model_key, random_state=cfg.training.seeds[0],
                        **kwargs)
        m.fit(Z_full, y)
        fitted[model_key] = m
        logger.info("refit %s on full data (params=%s)",
                    model_key, m.n_parameters())

    # conformal calibration from out-of-fold predictions of the primary
    # quantum model (honest coverage on held-out data)
    q_primary = next((k for k in QUANTUM_MODELS if k in result["models"]),
                     None)
    conformal = ConformalClassifier(alpha=0.1)
    if q_primary:
        pooled = result["models"][q_primary]["pooled_predictions"]
        y_pooled = np.array(pooled["y"])
        # split pooled OOF predictions: first 60% calibration, rest eval
        n_cal = int(0.6 * len(y_pooled))
        q_hat = conformal.calibrate(y_pooled[:n_cal],
                                    _as_proba2(np.array(pooled["p"])[:n_cal]))
        sets = conformal.predict_set(
            _as_proba2(np.array(pooled["p"])[n_cal:]))
        cov = coverage_stats(sets, y_pooled[n_cal:])
        result["conformal"] = {
            "model": q_primary, "alpha": 0.1, "q_hat": round(q_hat, 6),
            **cov,
            "note": "inductive conformal (LAC) on pooled out-of-fold "
                    "predictions; calibration split 60/40",
        }
        store.save_result(exp_id, result)

    # ---- F9 components ------------------------------------------------------
    # entanglement ablation on a stratified 70/30 split (parameter-matched)
    ablation = None
    abl_target = next((k for k in ("vqc", "reupload", "hybrid_qnn")
                       if k in result["models"]), None)
    if cfg.certificate.entanglement_ablation and abl_target:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y,
            random_state=cfg.training.seeds[0])
        abl_pipe = PreprocessPipeline(
            compression_method=cfg.compression.method,
            target_dim=cfg.compression.target_dim)
        Ztr = abl_pipe.fit_transform(X_tr, y_tr, feature_names=feature_cols)
        Zte = abl_pipe.transform(X_te)
        ablation = entanglement_ablation(
            abl_target, Ztr, y_tr, Zte, y_te,
            model_kwargs=engine._model_kwargs(abl_target, matched_params),
            seed=cfg.training.seeds[0])

    # dequantization: classical approximations of the QSVM quantum kernel
    dequant = None
    if (cfg.certificate.dequantization_rff or cfg.certificate.dequantization_nystrom) \
            and "qsvm" in result["models"]:
        from sklearn.model_selection import train_test_split
        from ..models.quantum.qsvm import QSVM
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y,
            random_state=cfg.training.seeds[0])
        dq_pipe = PreprocessPipeline(
            compression_method=cfg.compression.method,
            target_dim=cfg.compression.target_dim)
        Ztr = dq_pipe.fit_transform(X_tr, y_tr, feature_names=feature_cols)
        Zte = dq_pipe.transform(X_te)
        qsvm = QSVM(n_qubits=cfg.model.n_qubits,
                    encoding="zz" if cfg.model.encoding == "zz" else "angle",
                    backend=cfg.model.backend,
                    random_state=cfg.training.seeds[0])
        s_tr = qsvm._states(Ztr)
        s_te = qsvm._states(Zte)
        K_tr = np.real(np.abs(s_tr.conj() @ s_tr.T) ** 2)
        K_te = np.real(np.abs(s_te.conj() @ s_tr.T) ** 2)
        dequant = {"quantum_kernel_auroc": None}
        # quantum kernel reference on the same split
        qsvm2 = QSVM(n_qubits=cfg.model.n_qubits,
                     encoding=qsvm.encoding, backend=qsvm.backend,
                     random_state=cfg.training.seeds[0])
        qsvm2._train_states = s_tr
        from sklearn.svm import SVC
        svc = SVC(kernel="precomputed", C=1.0, probability=True,
                  random_state=cfg.training.seeds[0]).fit(K_tr, y_tr)
        p_q = svc.predict_proba(K_te)[:, 1]
        if len(np.unique(y_te)) == 2:
            from sklearn.metrics import roc_auc_score
            dequant["quantum_kernel_auroc"] = round(
                float(roc_auc_score(y_te, p_q)), 6)
        if cfg.certificate.dequantization_rff:
            dequant["rff"] = random_fourier_approximation(
                K_tr, y_tr, K_te, y_te, seed=cfg.training.seeds[0])
        if cfg.certificate.dequantization_nystrom:
            dequant["nystrom"] = nystrom_approximation(
                K_tr, y_tr, K_te, y_te, seed=cfg.training.seeds[0])

    # ---- F10: bottleneck honesty meter -----------------------------------------
    honesty = None
    honesty_target = abl_target or q_primary
    if cfg.certificate.bottleneck_honesty_meter and honesty_target:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y,
            random_state=cfg.training.seeds[0])
        hm_pipe = PreprocessPipeline(
            compression_method=cfg.compression.method,
            target_dim=cfg.compression.target_dim)
        hm_pipe.fit(X_tr, y_tr, feature_names=feature_cols)
        qm = build_model(honesty_target, random_state=cfg.training.seeds[0],
                         **engine._model_kwargs(honesty_target, matched_params))
        qm.fit(hm_pipe.transform(X_tr), y_tr)
        honesty = bottleneck_honesty_meter(
            hm_pipe, qm, X_tr, y_tr, X_te, y_te,
            seed=cfg.training.seeds[0])

    # ---- F12: cascade ------------------------------------------------------------
    cascade = None
    if cfg.cascade.enabled:
        screening_key = cfg.cascade.screening_model
        q_key = q_primary or "vqc"
        if screening_key in result["models"] and q_key in result["models"]:
            ps = np.array(
                result["models"][screening_key]["pooled_predictions"]["p"])
            pq = np.array(
                result["models"][q_key]["pooled_predictions"]["p"])
            y_pooled = np.array(
                result["models"][q_key]["pooled_predictions"]["y"])
            cascade = cascade_evaluate(
                ps, pq, y_pooled,
                band_low=cfg.cascade.low_confidence_min,
                band_high=cfg.cascade.low_confidence_max)
            cascade["configuration"]["screening_model"] = screening_key
            cascade["configuration"]["quantum_model"] = q_key

    _progress({"stage": "certificate_generation", "completed_fits": None})

    # ---- F9: certificate ------------------------------------------------------------
    cert = None
    if cfg.certificate.enabled:
        cert = build_certificate(
            result, ablation, dequant, honesty,
            config={
                "experiment_id": exp_id,
                "config_hash": cfg.config_hash(),
                "seeds": cfg.training.seeds,
            },
            reproducibility=repro)
        store.save_json(exp_id, "certificate", cert)
        store.save_text(exp_id, "certificate_report",
                        certificate_to_markdown(cert))

    result["ablation"] = ablation
    result["dequantization"] = dequant
    result["bottleneck_honesty_meter"] = honesty
    result["cascade"] = cascade
    result["conformal"] = result.get("conformal")
    store.save_result(exp_id, result)

    # ---- F8: explainability (on refit deployment artifacts) -------------------------
    if run_explainability:
        _progress({"stage": "explainability", "completed_fits": None})
        explain = _run_explainability(
            cfg, store, exp_id, pipe_full, fitted, X, y, feature_cols,
            q_primary, explain_samples)
        result["explainability_summary"] = explain["summary"]
        store.save_result(exp_id, result)

    # ---- persist deployment artifacts ------------------------------------------
    store.save_binary(exp_id, "pipeline", pipe_full)
    store.save_binary(exp_id, "models", fitted)
    store.save_binary(exp_id, "conformal", conformal)

    # ---- research report ------------------------------------------------------------
    report = _research_report(cfg, result, cert)
    store.save_text(exp_id, "report", report)

    # ---- MLflow mirror (optional) ----------------------------------------------------
    mlflow_ok = store.log_to_mlflow(
        exp_id, cfg, result,
        artifacts=[exp_dir / "result.json", exp_dir / "config.yaml"] +
        ([exp_dir / "certificate.json"] if cert else []))

    result["mlflow_logged"] = bool(mlflow_ok)
    result["total_runtime_s"] = round(time.time() - t_start, 2)
    store.save_result(exp_id, result)
    logger.info("Experiment %s complete in %.1fs",
                exp_id, result["total_runtime_s"])
    return result


def _as_proba2(p1: np.ndarray) -> np.ndarray:
    return np.column_stack([1 - p1, p1])


def _run_explainability(cfg, store, exp_id, pipe, fitted, X, y,
                        feature_cols, q_primary, explain_samples
                        ) -> Dict[str, Any]:
    """Explainability for the primary quantum model + a classical reference."""
    out: Dict[str, Any] = {}
    from sklearn.model_selection import train_test_split
    # stratified explain subset keeps both classes present
    n_exp = min(max(explain_samples, 12), len(X))
    X_exp, _, y_exp, _ = train_test_split(
        X, y, train_size=n_exp, stratify=y, random_state=42)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_exp), size=min(6, len(X_exp)), replace=False)

    for key in [q_primary, "xgboost", "logistic_regression"]:
        if key not in fitted:
            continue
        model = fitted[key]
        entry: Dict[str, Any] = {"model": key}

        entry["shap"] = shap_global_local(pipe, model, X, X_exp)
        if len(np.unique(y_exp)) == 2:
            entry["permutation"] = permutation_global(
                pipe, model, X_exp, y_exp, n_repeats=5)
        else:
            entry["permutation"] = {"available": False,
                                    "reason": "single-class explain subset"}
        entry["sensitivity"] = sensitivity_analysis(
            pipe, model, X, feature_cols)
        if key == q_primary:
            entry["quantum_diagnostics"] = quantum_diagnostics(
                model, pipe, X_exp, feature_cols)
            entry["counterfactual"] = _cf_for_sample(
                pipe, model, X_exp[idx[0]], feature_cols)
        out[key] = entry

    store.save_json(exp_id, "explainability", out)
    summary = {k: {
        "shap_available": v["shap"]["available"],
        "top_features_shap": _top_features(
            v["shap"].get("global_mean_abs_shap"), feature_cols),
        "top_features_permutation": _top_features(
            v["permutation"].get("importances_mean"), feature_cols),
        "quantum_diagnostics_available": bool(
            v.get("quantum_diagnostics", {}).get("available")),
    } for k, v in out.items()}
    return {"details": out, "summary": summary}


def _cf_for_sample(pipe, model, x_row, feature_names):
    from ..explain.explainer import counterfactual_search
    return counterfactual_search(pipe, model, x_row,
                                 feature_names=feature_names)


def _top_features(values, feature_names, k: int = 5):
    if not values:
        return []
    vals = list(map(float, values))
    order = np.argsort(vals)[::-1][:k]
    return [{"feature": feature_names[i], "value": round(vals[i], 6)}
            for i in order]


def _research_report(cfg, result, cert) -> str:
    """Markdown research report for the Research Reports view."""
    lines = [
        f"# Research Report — {cfg.experiment_id}",
        "",
        f"**Dataset:** {result['dataset']} "
        f"({result['n_samples']} samples, {result['n_features']} features)  ",
        f"**Protocol:** {result['protocol']['outer_folds']}-fold outer CV x "
        f"{result['protocol']['inner_folds']}-fold inner CV, "
        f"seeds {cfg.training.seeds}  ",
        f"**Compression:** {cfg.compression.method} -> "
        f"{cfg.compression.target_dim} dims  ",
        f"**Configuration hash:** `{cfg.config_hash()}`",
        "",
        "> RESEARCH AND EDUCATIONAL USE ONLY — not a medical device, not "
        "clinically validated, not for diagnosis.",
        "",
        "## Benchmark results (mean +/- std over folds x seeds)",
        "",
        "| Model | Family | AUROC | AUPRC | MCC | Brier | Params |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for k, m in sorted(result["models"].items(),
                       key=lambda kv: -(kv[1]["summary"]["auroc"]["mean"] or 0)):
        s = m["summary"]
        lines.append(
            f"| {k} | {m['family']} | "
            f"{s['auroc']['mean']:.4f} +/- {s['auroc']['std']:.4f} | "
            f"{s['auprc']['mean']:.4f} | {s['mcc']['mean']:.4f} | "
            f"{s['brier']['mean']:.4f} | {m['param_count']} |")

    if result.get("cascade"):
        c = result["cascade"]
        lines += [
            "",
            "## Second-opinion cascade (F12)",
            "",
            f"- Routed to quantum: {c['routing']['n_routed_to_quantum']}"
            f"/{c['routing']['n_total']} "
            f"({c['routing']['coverage_routed_fraction']:.1%})  ",
            f"- AUROC — screening: {c['performance']['auroc_screening_alone']:.4f}, "
            f"quantum: {c['performance']['auroc_quantum_alone']:.4f}, "
            f"cascade: {c['performance']['auroc_cascade']:.4f}",
        ]
    if result.get("bottleneck_honesty_meter"):
        h = result["bottleneck_honesty_meter"]
        lines += [
            "",
            "## Bottleneck honesty meter (F10)",
            "",
            f"- Compressed-space baseline AUROC: "
            f"{h['compressed_space_baseline']['auroc']:.4f}  ",
            f"- Quantum AUROC: {h['quantum_score']['auroc']:.4f}  ",
            f"- Gap: {h['performance_gap_quantum_minus_baseline']} — "
            f"{h['reading']}",
        ]
    if cert:
        lines += [
            "",
            "## Quantum Advantage Certificate (F9)",
            "",
            f"**Verdict: {cert['final_evidence_classification']}**  ",
            f"{cert['final_evidence_rationale']}",
            "",
            f"Full certificate: `experiments/runs/{cfg.experiment_id}/"
            "certificate.json`",
        ]
    lines += [
        "",
        "## Reproducibility",
        "",
        "```json",
        __import__("json").dumps(result.get("reproducibility", {}),
                                  indent=2, default=str),
        "```",
        "",
        f"Reproduce with: `python scripts/reproduce.py --config "
        f"configs/experiments/{cfg.dataset.name}.yaml`",
    ]
    return "\n".join(lines)
