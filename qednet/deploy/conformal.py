"""F6 Split Conformal Prediction."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split


class SplitConformalPredictor(BaseEstimator, ClassifierMixin):
    """
    Split Conformal Prediction for calibrated prediction sets.
    Returns prediction sets with guaranteed coverage.
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        preprocess_fn,
        calibration_size: float = 0.2,
        confidence: float = 0.9,
        random_state: int = 42,
    ):
        self.base_model = base_model
        self.preprocess_fn = preprocess_fn
        self.calibration_size = calibration_size
        self.confidence = confidence
        self.random_state = random_state
        self._model = None
        self._preprocess = None
        self._quantile = None
        self._fitted = False

    def fit(self, X, y):
        self._preprocess = self.preprocess_fn()

        # Split into train and calibration
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=self.calibration_size,
            random_state=self.random_state, stratify=y
        )

        X_train_trans = self._preprocess.fit_transform(X_train, y_train)
        X_cal_trans = self._preprocess.transform(X_cal)

        # Fit model on training
        self._model = clone(self.base_model)
        self._model.fit(X_train_trans, y_train)

        # Compute conformity scores on calibration set
        cal_proba = self._model.predict_proba(X_cal_trans)[:, 1]
        # Score = 1 - p_true for true class
        scores = 1 - cal_proba[y_cal == 1]
        scores = np.concatenate([scores, cal_proba[y_cal == 0]])  # for class 0: p_1

        # Quantile for coverage
        n_cal = len(scores)
        level = np.ceil((n_cal + 1) * self.confidence) / n_cal
        self._quantile = np.quantile(scores, min(level, 1.0))

        self._fitted = True
        self.classes_ = np.array([0, 1])
        return self

    def predict_set(self, X) -> np.ndarray:
        """
        Returns prediction sets:
        - 0: {0}
        - 1: {1}
        - 2: {0, 1} (ambiguous)
        """
        if not self._fitted:
            raise RuntimeError("Not fitted")

        X_trans = self._preprocess.transform(X)
        proba = self._model.predict_proba(X_trans)[:, 1]

        # p_1 >= 1 - quantile -> include class 1
        # p_0 >= 1 - quantile -> include class 0
        p1 = proba
        p0 = 1 - proba
        threshold = 1 - self._quantile

        include_1 = p1 >= threshold
        include_0 = p0 >= threshold

        pred_sets = np.zeros(len(X), dtype=int)
        pred_sets[include_1 & ~include_0] = 1
        pred_sets[~include_1 & include_0] = 0
        pred_sets[include_1 & include_0] = 2  # ambiguous

        return pred_sets

    def predict(self, X):
        """Single prediction (most probable class)."""
        sets = self.predict_set(X)
        # For ambiguous, pick higher prob
        X_trans = self._preprocess.transform(X)
        proba = self._model.predict_proba(X_trans)[:, 1]
        pred = (proba >= 0.5).astype(int)
        pred[sets == 2] = pred[sets == 2]  # keep most probable
        return pred

    def predict_proba(self, X):
        X_trans = self._preprocess.transform(X)
        return self._model.predict_proba(X_trans)

    def get_coverage_info(self, X, y) -> dict:
        """Evaluate empirical coverage."""
        sets = self.predict_set(X)
        covered = (sets == 2) | ((sets == 1) & (y == 1)) | ((sets == 0) & (y == 0))
        return {
            "empirical_coverage": float(np.mean(covered)),
            "target_coverage": self.confidence,
            "ambiguous_rate": float(np.mean(sets == 2)),
            "set_sizes": np.bincount(sets, minlength=3).tolist(),
        }

    def metadata(self) -> dict:
        return {
            "model": "split_conformal",
            "confidence": self.confidence,
            "calibration_size": self.calibration_size,
            "quantile": float(self._quantile) if self._quantile is not None else None,
        }