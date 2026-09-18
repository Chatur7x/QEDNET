"""Required evaluation metrics (F7 / Section 15).

AUROC, AUPRC, Sensitivity, Specificity, Sensitivity at 95% Specificity,
F1, MCC, Brier Score, ECE — plus timing and resource records collected by
the engine. All metrics are computed from real predictions only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score,
                             precision_recall_curve, roc_curve)


def expected_calibration_error(y_true, p_pos, n_bins: int = 10) -> float:
    """ECE with equal-width confidence bins."""
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p_pos), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= 0) & (p <= hi)
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / len(p)) * abs(conf - acc)
    return float(ece)


def sensitivity_at_specificity(y_true, p_pos, target_spec: float = 0.95
                               ) -> float:
    """Sensitivity at the operating point with specificity >= target.

    Thresholds are candidate cut points of the evaluated predictions
    themselves (no external data).
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p_pos)
    thresholds = np.unique(p)
    thresholds = np.concatenate([[np.inf], np.sort(thresholds)[::-1]])
    best_sens = 0.0
    for thr in thresholds:
        pred = p >= thr
        tp = int(np.sum(pred & (y_true == 1)))
        fn = int(np.sum(~pred & (y_true == 1)))
        tn = int(np.sum(~pred & (y_true == 0)))
        fp = int(np.sum(pred & (y_true == 0)))
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        if spec >= target_spec:
            best_sens = max(best_sens, sens)
    return float(best_sens)


def reliability_curve_data(y_true, p_pos, n_bins: int = 10):
    """Binned reliability (calibration) curve data."""
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p_pos), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= 0) & (p <= hi)
        if m.sum() == 0:
            continue
        centers.append(float(p[m].mean()))
        accs.append(float(y_true[m].mean()))
        counts.append(int(m.sum()))
    return centers, accs, counts


def compute_metrics(y_true: np.ndarray, p_pos: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Full metric suite for binary predictions with positive-class prob."""
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p_pos, dtype=float), 0, 1)
    pred = (p >= threshold).astype(int)

    has_both = len(np.unique(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)

    out: Dict[str, float] = {
        "auroc": float(roc_auc_score(y_true, p)) if has_both else float("nan"),
        "auprc": float(average_precision_score(y_true, p)) if has_both else float("nan"),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "sens_at_spec_95": sensitivity_at_specificity(y_true, p, 0.95)
        if has_both else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if has_both else float("nan"),
        "brier": float(brier_score_loss(y_true, p)),
        "ece": expected_calibration_error(y_true, p),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
    return out


def roc_curve_data(y_true, p_pos) -> Tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, p_pos)
    return fpr, tpr


def pr_curve_data(y_true, p_pos) -> Tuple[np.ndarray, np.ndarray]:
    prec, rec, _ = precision_recall_curve(y_true, p_pos)
    return prec, rec
