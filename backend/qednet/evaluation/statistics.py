"""Statistical analysis (Section 16).

DeLong test for paired AUROC comparison, bootstrap confidence intervals,
Nadeau-Bengio corrected CV comparison, and decision curve analysis.
No significance is ever claimed without the test actually being performed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# DeLong test (paired ROC curves on the same samples)
# ---------------------------------------------------------------------------

def _placements(pred_pos: np.ndarray, pred_neg: np.ndarray):
    """DeLong placement statistics.

    Returns (auc, v10, v01) where v10_i is positive i's placement among the
    negatives and v01_j is negative j's placement among the positives
    (ties count 0.5). This is the exact O(m*n) DeLong construction.
    """
    diff = pred_pos[:, None] - pred_neg[None, :]  # (m, n)
    wins = (diff > 0).astype(float) + 0.5 * (diff == 0)
    auc_val = float(wins.mean())
    v10 = wins.mean(axis=1)  # per positive
    v01 = wins.mean(axis=0)  # per negative
    return auc_val, v10, v01


def delong_test(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray
                ) -> Dict[str, float]:
    """DeLong paired test: AUC1 - AUC2 with covariance-corrected z-test."""
    y = np.asarray(y_true).astype(int)
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    if len(np.unique(y)) < 2:
        return {"delong_available": False, "reason": "single-class sample"}

    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        return {"delong_available": False, "reason": "too few class samples"}

    auc1, v10_1, v01_1 = _placements(p1[pos], p1[neg])
    auc2, v10_2, v01_2 = _placements(p2[pos], p2[neg])

    var1 = float(np.var(v10_1, ddof=1) / m + np.var(v01_1, ddof=1) / n)
    var2 = float(np.var(v10_2, ddof=1) / m + np.var(v01_2, ddof=1) / n)
    cov10 = float(np.cov(v10_1, v10_2, ddof=1)[0, 1] / m)
    cov01 = float(np.cov(v01_1, v01_2, ddof=1)[0, 1] / n)
    var_diff = var1 + var2 - 2.0 * (cov10 + cov01)

    diff = auc1 - auc2
    z = diff / np.sqrt(max(var_diff, 1e-18))
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return {
        "delong_available": True,
        "auc1": round(float(auc1), 6), "auc2": round(float(auc2), 6),
        "auc_diff": round(float(diff), 6),
        "z": round(float(z), 6), "p_value": float(p_value),
        "significant_005": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true: np.ndarray, p_pos: np.ndarray, metric: str = "auroc",
                 n_resamples: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> Dict[str, float]:
    """Percentile bootstrap CI for a metric (default 2000 resamples)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(p_pos)
    n = len(y)

    def metric_fn(yy, pp):
        if len(np.unique(yy)) < 2:
            return np.nan
        if metric == "auroc":
            return roc_auc_score(yy, pp)
        if metric == "auprc":
            from sklearn.metrics import average_precision_score
            return average_precision_score(yy, pp)
        if metric == "brier":
            from sklearn.metrics import brier_score_loss
            return brier_score_loss(yy, np.clip(pp, 0, 1))
        raise ValueError(f"unknown bootstrap metric {metric}")

    point = metric_fn(y, p)
    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats[b] = metric_fn(y[idx], p[idx])
    stats = stats[~np.isnan(stats)]
    lo = float(np.percentile(stats, 100 * alpha / 2)) if len(stats) else np.nan
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2))) if len(stats) else np.nan
    return {
        "metric": metric, "point": float(point) if not np.isnan(point) else None,
        "ci_low": lo, "ci_high": hi, "n_resamples": int(n_resamples),
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Corrected cross-validation comparison (Nadeau & Bengio 2003)
# ---------------------------------------------------------------------------

def corrected_cv_ttest(scores_a: List[float], scores_b: List[float],
                       n_train: int, n_test: int,
                       k_folds: int) -> Dict[str, float]:
    """Corrected resampled t-test for CV score differences."""
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    if len(a) != len(b) or len(a) < 2:
        return {"available": False, "reason": "need aligned score lists"}
    d = a - b
    mean_d = float(d.mean())
    # correction factor: n_test/n_train ratio + fold overlap
    ntf = n_test / n_train
    factor = 1.0 / k_folds + ntf
    var_d = float(np.var(d, ddof=1))
    t = mean_d / np.sqrt(max(factor * var_d, 1e-18))
    from scipy.stats import t as t_dist
    dof = len(d) - 1
    p_value = 2 * (1 - t_dist.cdf(abs(t), dof))
    return {
        "available": True,
        "mean_diff": round(mean_d, 6),
        "t": round(float(t), 6), "p_value": float(p_value),
        "significant_005": bool(p_value < 0.05),
        "correction": "Nadeau-Bengio corrected resampled t-test",
        "k_folds": int(k_folds), "n_train": int(n_train), "n_test": int(n_test),
    }


# ---------------------------------------------------------------------------
# Decision curve analysis (net benefit)
# ---------------------------------------------------------------------------

def decision_curve_data(y_true: np.ndarray, p_pos: np.ndarray,
                        thresholds: np.ndarray | None = None
                        ) -> Dict[str, Any]:
    """Net benefit across threshold probabilities (treat-all vs treat-none)."""
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p_pos, float), 0, 1)
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    n = len(y)
    prev = y.mean()
    rows = []
    for thr in thresholds:
        pred = p >= thr
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        nb_model = tp / n - fp / n * (thr / max(1e-9, 1 - thr))
        nb_all = prev - (1 - prev) * (thr / max(1e-9, 1 - thr))
        rows.append({
            "threshold": round(float(thr), 2),
            "net_benefit_model": round(float(nb_model), 6),
            "net_benefit_treat_all": round(float(nb_all), 6),
            "net_benefit_treat_none": 0.0,
        })
    return {"thresholds": [r["threshold"] for r in rows],
            "model": [r["net_benefit_model"] for r in rows],
            "treat_all": [r["net_benefit_treat_all"] for r in rows],
            "rows": rows}
