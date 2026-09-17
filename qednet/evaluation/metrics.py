"""Evaluation metrics for binary classification."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


def auroc(y_true: NDArray, y_proba: NDArray) -> float:
    """Area Under the ROC Curve."""
    return roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)


def auprc(y_true: NDArray, y_proba: NDArray) -> float:
    """Area Under the Precision-Recall Curve."""
    return average_precision_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)


def sensitivity_at_specificity(
    y_true: NDArray,
    y_proba: NDArray,
    target_specificity: float = 0.95,
) -> float:
    """Sensitivity at a fixed specificity threshold."""
    y_score = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    idx = np.where(specificity >= target_specificity)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def mcc(y_true: NDArray, y_proba: NDArray, threshold: float = 0.5) -> float:
    """Matthews Correlation Coefficient."""
    y_pred = (y_proba[:, 1] if y_proba.ndim > 1 else y_proba) >= threshold
    return matthews_corrcoef(y_true, y_pred)


def brier_score(y_true: NDArray, y_proba: NDArray) -> float:
    """Brier Score."""
    y_score = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    return brier_score_loss(y_true, y_score)


def ece(y_true: NDArray, y_proba: NDArray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    y_score = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    return np.mean(np.abs(prob_true - prob_pred))


def bootstrap_confidence_interval(
    y_true: NDArray,
    y_proba: NDArray,
    metric_func,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            score = metric_func(y_true[idx], y_proba[idx])
            scores.append(score)
        except Exception:
            continue

    if len(scores) < 10:
        return (np.nan, np.nan)

    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return (float(lower), float(upper))


def delong_test(
    y_true: NDArray,
    y_proba1: NDArray,
    y_proba2: NDArray,
) -> dict[str, float]:
    """DeLong test for comparing two AUROCs.

    Returns:
        Dictionary with 'p_value' and 'auroc_diff'
    """
    # Simplified implementation using bootstrap
    # For production, use proper DeLong implementation
    from scipy.stats import norm

    n = len(y_true)
    auroc1 = auroc(y_true, y_proba1)
    auroc2 = auroc(y_true, y_proba2)

    # Bootstrap variance estimation
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(1000):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            d = auroc(y_true[idx], y_proba1[idx]) - auroc(y_true[idx], y_proba2[idx])
            diffs.append(d)
        except Exception:
            continue

    if len(diffs) < 10:
        return {"p_value": np.nan, "auroc_diff": auroc1 - auroc2}

    diff_mean = np.mean(diffs)
    diff_std = np.std(diffs)
    if diff_std > 0:
        z = abs(auroc1 - auroc2) / diff_std
        p = 2 * (1 - norm.cdf(z))
    else:
        p = 1.0

    return {"p_value": float(p), "auroc_diff": float(auroc1 - auroc2)}


def decision_curve_analysis(
    y_true: NDArray,
    y_proba: NDArray,
    thresholds: Optional[NDArray] = None,
) -> dict[str, NDArray]:
    """Decision curve analysis for clinical utility."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    y_score = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    n = len(y_true)
    prevalence = np.mean(y_true)

    net_benefit = []
    treat_all = []
    treat_none = np.zeros_like(thresholds)

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        nb = tp / n - fp / n * (t / (1 - t))
        net_benefit.append(nb)

        ta = prevalence - (1 - prevalence) * (t / (1 - t))
        treat_all.append(ta)

    return {
        "thresholds": thresholds,
        "net_benefit": np.array(net_benefit),
        "treat_all": np.array(treat_all),
        "treat_none": treat_none,
    }


def compute_all_metrics(
    y_true: NDArray,
    y_proba: NDArray,
    metrics: list[str],
    bootstrap_ci: int = 0,
) -> dict[str, Any]:
    """Compute all specified metrics."""
    results = {}

    metric_funcs = {
        "auroc": auroc,
        "auprc": auprc,
        "sens_at_spec95": lambda y, p: sensitivity_at_specificity(y, p, 0.95),
        "mcc": mcc,
        "brier": brier_score,
        "ece": ece,
    }

    for name in metrics:
        if name in metric_funcs:
            try:
                value = metric_funcs[name](y_true, y_proba)
                results[name] = value

                if bootstrap_ci > 0:
                    ci = bootstrap_confidence_interval(
                        y_true, y_proba, metric_funcs[name], n_bootstrap=bootstrap_ci
                    )
                    results[f"{name}_ci"] = ci
            except Exception as e:
                results[name] = np.nan
                if bootstrap_ci > 0:
                    results[f"{name}_ci"] = (np.nan, np.nan)

    return results


# Import roc_curve here to avoid circular import
from sklearn.metrics import roc_curve