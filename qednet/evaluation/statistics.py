"""Statistical testing and confidence intervals (D8)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats
from scipy.stats import norm


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point = metric_fn(y_true, y_prob)
    bootstraps = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            bootstraps.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            continue

    if len(bootstraps) < 10:
        return point, point, point

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(bootstraps, 100 * alpha))
    upper = float(np.percentile(bootstraps, 100 * (1 - alpha)))
    return point, lower, upper


def delong_roc_test(y_true: np.ndarray, y_prob1: np.ndarray, y_prob2: np.ndarray) -> dict:
    """
    DeLong test for comparing two correlated ROC AUCs.
    Returns p-value and test statistic.
    """

    # Implementation based on DeLong et al. 1988
    def _delong_covariance(y_true, y_prob_a, y_prob_b):
        # Compute the covariance matrix of AUCs
        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)

        # Get indices
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]

        # V10 and V01 for each model
        v10_a = np.zeros(n1)
        v01_a = np.zeros(n0)
        v10_b = np.zeros(n1)
        v01_b = np.zeros(n0)

        for i, p in enumerate(pos_idx):
            v10_a[i] = np.mean(y_prob_a[p] > y_prob_a[neg_idx]) + 0.5 * np.mean(
                y_prob_a[p] == y_prob_a[neg_idx]
            )
            v10_b[i] = np.mean(y_prob_b[p] > y_prob_b[neg_idx]) + 0.5 * np.mean(
                y_prob_b[p] == y_prob_b[neg_idx]
            )

        for i, n in enumerate(neg_idx):
            v01_a[i] = np.mean(y_prob_a[pos_idx] > y_prob_a[n]) + 0.5 * np.mean(
                y_prob_a[pos_idx] == y_prob_a[n]
            )
            v01_b[i] = np.mean(y_prob_b[pos_idx] > y_prob_b[n]) + 0.5 * np.mean(
                y_prob_b[pos_idx] == y_prob_b[n]
            )

        # Covariances
        cov_10 = np.cov(v10_a, v10_b)[0, 1]
        cov_01 = np.cov(v01_a, v01_b)[0, 1]

        var_a = np.var(v10_a) / n1 + np.var(v01_a) / n0
        var_b = np.var(v10_b) / n1 + np.var(v01_b) / n0
        cov = cov_10 / n1 + cov_01 / n0

        return var_a, var_b, cov

    try:
        var_a, var_b, cov = _delong_covariance(y_true, y_prob1, y_prob2)

        auc1 = np.mean(
            y_prob1[y_true == 1][:, None] > y_prob1[y_true == 0][None, :]
        ) + 0.5 * np.mean(y_prob1[y_true == 1][:, None] == y_prob1[y_true == 0][None, :])
        auc2 = np.mean(
            y_prob2[y_true == 1][:, None] > y_prob2[y_true == 0][None, :]
        ) + 0.5 * np.mean(y_prob2[y_true == 1][:, None] == y_prob2[y_true == 0][None, :])

        diff = auc1 - auc2
        var_diff = var_a + var_b - 2 * cov
        se = np.sqrt(max(var_diff, 1e-10))
        z = diff / se
        p = 2 * (1 - norm.cdf(abs(z)))

        return {
            "auc1": float(auc1),
            "auc2": float(auc2),
            "diff": float(diff),
            "z": float(z),
            "p_value": float(p),
            "significant": p < 0.05,
        }
    except Exception:
        return {
            "auc1": 0.0,
            "auc2": 0.0,
            "diff": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "significant": False,
        }


def paired_t_test(metric1: np.ndarray, metric2: np.ndarray) -> dict:
    """Paired t-test for comparing metrics across folds/seeds."""
    diff = metric1 - metric2
    n = len(diff)
    if n < 2:
        return {"t": 0.0, "p": 1.0, "mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return {
            "t": 0.0,
            "p": 1.0,
            "mean_diff": float(mean_diff),
            "ci_lower": float(mean_diff),
            "ci_upper": float(mean_diff),
        }
    t = mean_diff / (std_diff / np.sqrt(n))
    p = 2 * (1 - stats.t.cdf(abs(t), n - 1))
    se = std_diff / np.sqrt(n)
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se
    return {
        "t": float(t),
        "p": float(p),
        "mean_diff": float(mean_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": p < 0.05,
    }


def corrected_paired_t_test(
    metric1: np.ndarray,
    metric2: np.ndarray,
    n_train: int,
    n_test: int,
) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio 2003) for cross-validation.
    Accounts for overlap in training sets.
    """
    n = len(metric1)
    diff = metric1 - metric2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return {
            "t": 0.0,
            "p": 1.0,
            "mean_diff": float(mean_diff),
            "ci_lower": float(mean_diff),
            "ci_upper": float(mean_diff),
        }
    # Correction factor for CV
    rho = n_test / (n_train + n_test)
    correction = np.sqrt((1 / n) + (rho / (1 - rho)))
    t = mean_diff / (std_diff * correction)
    p = 2 * (1 - stats.t.cdf(abs(t), n - 1))
    se = std_diff * correction
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se
    return {
        "t": float(t),
        "p": float(p),
        "mean_diff": float(mean_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": p < 0.05,
        "correction_factor": float(correction),
    }
