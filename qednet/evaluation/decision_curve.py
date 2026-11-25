"""Decision curve analysis (D8)."""

from __future__ import annotations

import numpy as np


def decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Compute decision curve analysis."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    n = len(y_true)
    prevalence = np.mean(y_true)

    net_benefits = []
    treat_all_nbs = []
    treat_none_nbs = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefits.append(nb)

        # Treat all
        treat_all = prevalence - (1 - prevalence) * (t / (1 - t))
        treat_all_nbs.append(treat_all)

        # Treat none
        treat_none_nbs.append(0.0)

    return {
        "thresholds": thresholds.tolist(),
        "net_benefit": net_benefits,
        "treat_all": treat_all_nbs,
        "treat_none": treat_none_nbs,
    }


def net_benefit_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    """Net benefit at a single threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    return float((tp / n) - (fp / n) * (threshold / (1 - threshold)))


def clinical_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_range: tuple = (0.05, 0.5),
) -> dict:
    """Clinical net benefit summary."""
    thresholds = np.linspace(threshold_range[0], threshold_range[1], 50)
    curve = decision_curve(y_true, y_prob, thresholds)

    # Area where model beats treat-all and treat-none
    nb = np.array(curve["net_benefit"])
    treat_all = np.array(curve["treat_all"])
    treat_none = np.array(curve["treat_none"])

    better_than_all = np.sum(np.maximum(nb - treat_all, 0))
    better_than_none = np.sum(np.maximum(nb - treat_none, 0))

    return {
        "thresholds": thresholds.tolist(),
        "net_benefit": nb.tolist(),
        "treat_all": treat_all.tolist(),
        "better_than_treat_all_auc": float(better_than_all),
        "better_than_treat_none_auc": float(better_than_none),
    }
