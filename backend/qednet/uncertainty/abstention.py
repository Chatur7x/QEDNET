"""F11 — Uncertainty & Abstention.

Calibrated probabilities, prediction uncertainty, inductive conformal
prediction (LAC sets) and abstention rules. The system is ALLOWED to say
"prediction uncertain — abstention triggered" and never forces a prediction
when the configured abstention rule says otherwise.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Inductive conformal prediction (LAC: least-ambiguous set-valued classifier)
# ---------------------------------------------------------------------------

class ConformalClassifier:
    """Split-conformal prediction sets at a target error level alpha.

    Calibration: nonconformity score s_i = 1 - p_i[y_i] on calibration data.
    Inference: class set = {y : p[y] >= 1 - q_hat} with q_hat the empirical
    (1-alpha) quantile of scores. Guarantees marginal coverage 1-alpha under
    exchangeability.
    """

    def __init__(self, alpha: float = 0.1):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.q_hat: Optional[float] = None

    def calibrate(self, y_cal: np.ndarray, p_cal: np.ndarray) -> float:
        y = np.asarray(y_cal, int)
        p = np.clip(np.asarray(p_cal, float), 0, 1)
        scores = 1.0 - p[np.arange(len(y)), y]
        n = len(scores)
        # finite-sample corrected quantile (ceil((n+1)(1-alpha)) / n)
        k = min(n, int(np.ceil((n + 1) * (1 - self.alpha))))
        order = np.sort(scores)
        self.q_hat = float(order[k - 1]) if k >= 1 else 1.0
        return self.q_hat

    def predict_set(self, p_new: np.ndarray) -> List[List[int]]:
        if self.q_hat is None:
            raise RuntimeError("Conformal classifier not calibrated")
        p = np.clip(np.asarray(p_new, float), 0, 1)
        thr = 1.0 - self.q_hat
        sets = []
        for row in p:
            sets.append([j for j in range(row.shape[0]) if row[j] >= thr])
        return sets


def coverage_stats(sets: List[List[int]], y_true: np.ndarray) -> Dict[str, float]:
    """Empirical coverage and set-size distribution."""
    y = np.asarray(y_true, int)
    cover = float(np.mean([y[i] in s for i, s in enumerate(sets)]))
    sizes = np.array([len(s) for s in sets])
    return {
        "empirical_coverage": cover,
        "mean_set_size": float(sizes.mean()),
        "frac_singleton": float(np.mean(sizes == 1)),
        "frac_empty": float(np.mean(sizes == 0)),
    }


# ---------------------------------------------------------------------------
# Uncertainty estimates
# ---------------------------------------------------------------------------

def predictive_entropy(p_pos: np.ndarray) -> np.ndarray:
    """Binary entropy of the positive-class probability."""
    p = np.clip(np.asarray(p_pos, float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def ensemble_uncertainty(p_list: List[np.ndarray]) -> np.ndarray:
    """Std of positive-class probability across an ensemble (e.g. seeds)."""
    P = np.stack([np.asarray(p, float) for p in p_list])
    return P.std(axis=0)


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

class AbstentionRule:
    """Configured abstention policy (never bypassed at prediction time).

    A prediction is ABSTAINED when any of the configured conditions hold:
      * max class probability < ``min_confidence`` (uninformative)
      * positive probability inside the ambiguity band
        [``band_low``, ``band_high``] (if enabled)
      * conformal prediction set size != 1 (if a conformal calibrator is set)
    """

    def __init__(self, min_confidence: float = 0.5,
                 band_low: float = 0.35, band_high: float = 0.65,
                 use_band: bool = True,
                 conformal: Optional[ConformalClassifier] = None):
        self.min_confidence = float(min_confidence)
        self.band_low = float(band_low)
        self.band_high = float(band_high)
        self.use_band = use_band
        self.conformal = conformal

    def decide(self, p_pos: np.ndarray) -> Dict[str, Any]:
        """Vectorised abstention decision + reasons."""
        p = np.clip(np.asarray(p_pos, float), 0, 1)
        max_p = np.maximum(p, 1 - p)
        uninformative = max_p < self.min_confidence
        ambiguous = np.zeros_like(uninformative)
        if self.use_band:
            ambiguous = (p >= self.band_low) & (p <= self.band_high)
        non_singleton = np.zeros_like(uninformative)
        if self.conformal is not None and self.conformal.q_hat is not None:
            thr = 1.0 - self.conformal.q_hat
            sets_ok = (p >= thr) | ((1 - p) >= thr)
            non_singleton = ~sets_ok
        abstain = uninformative | ambiguous | non_singleton
        return {
            "abstain": abstain,
            "reasons": {
                "uninformative": uninformative,
                "ambiguous_band": ambiguous,
                "conformal_non_singleton": non_singleton,
            },
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "min_confidence": self.min_confidence,
            "ambiguity_band": [self.band_low, self.band_high]
            if self.use_band else None,
            "conformal_alpha": self.conformal.alpha
            if self.conformal else None,
            "conformal_q_hat": self.conformal.q_hat
            if self.conformal else None,
        }
