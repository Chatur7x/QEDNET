"""Calibration (Section 17): Platt scaling and isotonic regression.

Reliability curves, Brier score and ECE are reported before/after
calibration. Calibrators are fit on training-fold predictions only.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .metrics import (expected_calibration_error, reliability_curve_data)


class PlattScaler:
    """Platt scaling: p_cal = sigmoid(a * logit(p) + b) via LR on logit."""

    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, p_pos: np.ndarray, y: np.ndarray) -> "PlattScaler":
        p = np.clip(np.asarray(p_pos, float), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        self.lr.fit(logit, np.asarray(y))
        return self

    def transform(self, p_pos: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p_pos, float), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return self.lr.predict_proba(logit)[:, 1]


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p_pos: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.ir.fit(np.asarray(p_pos, float), np.asarray(y, float))
        return self

    def transform(self, p_pos: np.ndarray) -> np.ndarray:
        return np.asarray(self.ir.predict(np.asarray(p_pos, float)), float)


def calibration_report(y_true: np.ndarray, p_raw: np.ndarray,
                       p_cal: np.ndarray) -> Dict[str, object]:
    """Reliability + Brier + ECE before/after calibration."""
    centers_raw, accs_raw, counts = reliability_curve_data(y_true, p_raw)
    centers_cal, accs_cal, _ = reliability_curve_data(y_true, p_cal)
    from sklearn.metrics import brier_score_loss
    return {
        "reliability_raw": {"centers": centers_raw, "accuracy": accs_raw,
                            "counts": counts},
        "reliability_calibrated": {"centers": centers_cal,
                                   "accuracy": accs_cal},
        "brier_raw": float(brier_score_loss(y_true, np.clip(p_raw, 0, 1))),
        "brier_calibrated": float(brier_score_loss(y_true, np.clip(p_cal, 0, 1))),
        "ece_raw": expected_calibration_error(y_true, p_raw),
        "ece_calibrated": expected_calibration_error(y_true, p_cal),
        "note": "calibration fitted on training-fold predictions only",
    }


def fit_calibrator(method: str, p_train: np.ndarray, y_train: np.ndarray):
    """Build a calibrator of the configured type."""
    if method == "platt":
        return PlattScaler().fit(p_train, y_train)
    if method == "isotonic":
        return IsotonicCalibrator().fit(p_train, y_train)
    return None
