"""Probability calibration (D8)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted


class PlattCalibrator(BaseEstimator, ClassifierMixin):
    """Platt scaling calibration."""

    def __init__(self, base_model=None):
        self.base_model = base_model
        self._calibrator = None
        self._fitted = False

    def fit(self, X, y):
        if self.base_model is not None:
            self._calibrator = CalibratedClassifierCV(
                clone(self.base_model), method="sigmoid", cv="prefit"
            )
            self.base_model.fit(X, y)
            self._calibrator.fit(X, y)
        else:
            # Fit logistic regression on decision scores
            from sklearn.linear_model import LogisticRegression

            self._calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
            self._calibrator.fit(X, y)
        self._fitted = True
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_calibrator")
        return self._calibrator.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class IsotonicCalibrator(BaseEstimator, ClassifierMixin):
    """Isotonic regression calibration."""

    def __init__(self, base_model=None):
        self.base_model = base_model
        self._calibrator = None
        self._fitted = False

    def fit(self, X, y):
        if self.base_model is not None:
            self._calibrator = CalibratedClassifierCV(
                clone(self.base_model), method="isotonic", cv="prefit"
            )
            self.base_model.fit(X, y)
            self._calibrator.fit(X, y)
        else:
            from sklearn.isotonic import IsotonicRegression

            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(X, y)
        self._fitted = True
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_calibrator")
        p1 = self._calibrator.predict(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    method: str = "platt",
    cv: int = 3,
) -> BaseEstimator:
    """Calibrate a fitted model's probabilities."""
    if method == "platt":
        return CalibratedClassifierCV(model, method="sigmoid", cv=cv).fit(X, y)
    elif method == "isotonic":
        return CalibratedClassifierCV(model, method="isotonic", cv=cv).fit(X, y)
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Evaluate calibration quality."""
    from qednet.evaluation.metrics import expected_calibration_error

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    prob_pred, prob_true, bin_counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        bin_counts.append(int(np.sum(m)))
        if np.any(m):
            prob_true.append(float(np.mean(y_true[m])))
            prob_pred.append(float(np.mean(y_prob[m])))
        else:
            prob_true.append(0.0)
            prob_pred.append(float((edges[b] + edges[b + 1]) / 2))

    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    mce = float(np.max(np.abs(np.array(prob_true) - np.array(prob_pred))))

    return {
        "ece": float(ece),
        "mce": mce,
        "curve": {
            "prob_pred": prob_pred,
            "prob_true": prob_true,
            "bin_counts": bin_counts,
        },
    }
