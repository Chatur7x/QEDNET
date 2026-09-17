"""N5 Shot-Noise Uncertainty."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def shot_noise_variance(p: float, shots: int) -> float:
    """Variance of probability estimate from finite shots: p(1-p)/S."""
    return p * (1 - p) / shots


def shot_noise_ci(p: float, shots: int, confidence: float = 0.95) -> tuple[float, float]:
    """Confidence interval for shot-noise limited probability."""
    var = shot_noise_variance(p, shots)
    se = np.sqrt(var)
    z = norm.ppf((1 + confidence) / 2)
    margin = z * se
    return max(0, p - margin), min(1, p + margin)


def propagate_shot_uncertainty(
    probs: np.ndarray,
    shots: int,
    confidence: float = 0.95,
) -> dict:
    """Compute uncertainty intervals for array of probabilities."""
    lower, upper = [], []
    for p in probs:
        l, u = shot_noise_ci(p, shots, confidence)
        lower.append(l)
        upper.append(u)
    return {
        "probabilities": probs.tolist(),
        "ci_lower": lower,
        "ci_upper": upper,
        "variances": [shot_noise_variance(p, shots) for p in probs],
    }


def combine_uncertainties(
    model_uncertainty: np.ndarray,  # e.g., from ensemble/bootstrap
    shot_uncertainty: np.ndarray,   # from shot noise
    weights: tuple = (1.0, 1.0),
) -> np.ndarray:
    """Combine model and shot uncertainty."""
    w1, w2 = weights
    return np.sqrt(w1 * model_uncertainty**2 + w2 * shot_uncertainty**2)


def evaluate_uncertainty_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_uncertainty: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Evaluate if uncertainty correlates with error."""
    errors = np.abs(y_prob - y_true)
    # Bin by uncertainty
    bins = np.percentile(y_uncertainty, np.linspace(0, 100, n_bins + 1))
    bin_errors = []
    bin_uncerts = []

    for i in range(n_bins):
        mask = (y_uncertainty >= bins[i]) & (y_uncertainty <= bins[i + 1])
        if np.sum(mask) > 0:
            bin_errors.append(np.mean(errors[mask]))
            bin_uncerts.append(np.mean(y_uncertainty[mask]))

    # Correlation
    if len(bin_errors) > 1:
        corr = np.corrcoef(bin_errors, bin_uncerts)[0, 1]
    else:
        corr = 0.0

    return {
        "bin_errors": bin_errors,
        "bin_uncertainties": bin_uncerts,
        "correlation": float(corr),
        "mean_error": float(np.mean(errors)),
        "mean_uncertainty": float(np.mean(y_uncertainty)),
    }