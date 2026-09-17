"""Evaluation metrics (D8)."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    brier_score_loss, roc_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve


def sensitivity_at_specificity(y_true, y_prob, target_spec: float = 0.95) -> float:
    """Sensitivity at fixed specificity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def specificity_at_sensitivity(y_true, y_prob, target_sens: float = 0.95) -> float:
    """Specificity at fixed sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(tpr >= target_sens)[0]
    if len(idx) == 0:
        return 0.0
    return float(1 - fpr[idx[0]])


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE); correct for empty bins / constant output."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if np.any(m):
            ece += m.mean() * abs(float(np.mean(y_true[m])) - float(np.mean(y_prob[m])))
    return float(ece)


def net_benefit(y_true, y_prob, threshold: float) -> float:
    """Decision curve net benefit at threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))


def decision_curve_auc(y_true, y_prob, thresholds: np.ndarray | None = None) -> float:
    """Area under the decision curve."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    nbs = [net_benefit(y_true, y_prob, t) for t in thresholds]
    # Net benefit of treating all vs treating none
    treat_all = np.mean(y_true) - (1 - np.mean(y_true)) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)
    # AUC of net benefit above treat-all and treat-none
    nb_array = np.array(nbs)
    diff_all = np.maximum(nb_array - treat_all, 0)
    diff_none = np.maximum(nb_array - treat_none, 0)
    return float(np.trapezoid(diff_all, thresholds) + np.trapezoid(diff_none, thresholds))


def compute_all_metrics(y_true, y_prob, y_pred=None) -> dict:
    """Compute all standard metrics."""
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "sens_at_spec95": sensitivity_at_specificity(y_true, y_prob, 0.95),
        "spec_at_sens95": specificity_at_sensitivity(y_true, y_prob, 0.95),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "decision_curve_auc": decision_curve_auc(y_true, y_prob),
    }