"""Conformal Prediction (F6)."""

from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray

from qednet.models.base import BaseEstimator


@dataclass
class ConformalResult:
    """Result from conformal prediction."""

    prediction_set: list[int]
    abstained: bool
    alpha: float
    calibration_coverage: float
    nonconformity_scores: NDArray | None = None
    threshold: float | None = None


class ConformalPredictor:
    """Model-agnostic conformal predictor for binary classification.

    Can wrap any model with fit/predict_proba interface.
    Provides finite-sample coverage guarantees under exchangeability.

    Example:
        predictor = ConformalPredictor(model, alpha=0.10)
        predictor.calibrate(X_cal, y_cal)
        result = predictor.predict(X_test)
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.10,
        score_type: str = "max_proba",
        random_state: int = 42,
    ):
        """Initialize conformal predictor.

        Args:
            model: Fitted base model with predict_proba method
            alpha: Miscoverage level (1 - alpha = target coverage)
            score_type: Nonconformity score type ('max_proba', 'margin', 'aps')
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if score_type not in {"max_proba", "margin", "aps"}:
            raise ValueError(f"Unknown score_type: {score_type}")

        self.model = model
        self.alpha = alpha
        self.score_type = score_type
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.calibration_scores: NDArray | None = None
        self.threshold: float | None = None
        self.calibration_coverage: float | None = None
        self._is_calibrated = False

    def _compute_scores(self, y_true: NDArray, y_proba: NDArray) -> NDArray:
        """Compute nonconformity scores."""
        y_true = np.asarray(y_true)
        y_proba = self._validate_probabilities(y_proba)
        if y_true.ndim != 1 or len(y_true) != len(y_proba):
            raise ValueError("y_true must be one binary label per probability row")
        if not np.isin(y_true, (0, 1)).all():
            raise ValueError("ConformalPredictor supports binary labels 0 and 1")
        y_true = y_true.astype(int)

        if self.score_type == "max_proba":
            true_class_proba = y_proba[np.arange(len(y_true)), y_true]
            return 1 - true_class_proba
        elif self.score_type == "margin":
            true_class_proba = y_proba[np.arange(len(y_true)), y_true]
            other_class_proba = y_proba[np.arange(len(y_true)), 1 - y_true]
            return 1 - (true_class_proba - other_class_proba)
        elif self.score_type == "aps":
            scores = np.empty(len(y_true), dtype=float)
            for i, label in enumerate(y_true):
                order = np.argsort(-y_proba[i], kind="stable")
                rank = int(np.flatnonzero(order == label)[0])
                mass_before = y_proba[i, order[:rank]].sum()
                scores[i] = mass_before + self._rng.random() * y_proba[i, label]
            return scores

    @staticmethod
    def _validate_probabilities(y_proba: NDArray) -> NDArray:
        probabilities = np.asarray(y_proba, dtype=float)
        if (
            probabilities.ndim != 2
            or probabilities.shape[1] != 2
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1.0)
        ):
            raise ValueError(
                "predict_proba must return finite binary probability rows summing to one"
            )
        return probabilities

    def calibrate(
        self,
        X_cal: NDArray,
        y_cal: NDArray,
    ) -> "ConformalPredictor":
        """Calibrate on held-out calibration set.

        Args:
            X_cal: Calibration features
            y_cal: Calibration labels

        Returns:
            self
        """
        if len(X_cal) != len(y_cal):
            raise ValueError("X_cal and y_cal must have same length")
        if len(X_cal) == 0:
            raise ValueError("Calibration set cannot be empty")

        self._rng = np.random.default_rng(self.random_state)
        y_cal = np.asarray(y_cal)
        y_proba = self._validate_probabilities(self.model.predict_proba(X_cal))
        self.calibration_scores = self._compute_scores(y_cal, y_proba)

        n_cal = len(self.calibration_scores)
        k = ceil((n_cal + 1) * (1 - self.alpha))
        if k > n_cal:
            self.threshold = 2.0 if self.score_type == "margin" else 1.0
        else:
            self.threshold = float(np.partition(self.calibration_scores, k - 1)[k - 1])
        self._is_calibrated = True
        self.calibration_coverage = float(np.mean(self.calibration_scores <= self.threshold))

        return self

    def _get_prediction_sets(self, y_proba: NDArray) -> list[list[int]]:
        """Get prediction sets for given probabilities."""
        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        n = len(y_proba)
        pred_sets = []

        for i in range(n):
            proba = y_proba[i]

            if self.score_type == "max_proba":
                score_0 = 1 - proba[0]
                score_1 = 1 - proba[1]
            elif self.score_type == "margin":
                score_0 = 1 - (proba[0] - proba[1])
                score_1 = 1 - (proba[1] - proba[0])
            else:  # randomized adaptive prediction sets
                scores = []
                order = np.argsort(-proba, kind="stable")
                for candidate in (0, 1):
                    rank = int(np.flatnonzero(order == candidate)[0])
                    mass_before = proba[order[:rank]].sum()
                    scores.append(mass_before + self._rng.random() * proba[candidate])
                score_0, score_1 = scores

            scores = [score_0, score_1]
            pred_set = [c for c, s in enumerate(scores) if s <= self.threshold]
            pred_sets.append(pred_set)

        return pred_sets

    def predict(self, X: NDArray) -> list[ConformalResult]:
        """Generate conformal prediction sets."""
        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        y_proba = self._validate_probabilities(self.model.predict_proba(X))
        pred_sets = self._get_prediction_sets(y_proba)

        results = []
        for pred_set in pred_sets:
            abstained = len(pred_set) == 2
            results.append(
                ConformalResult(
                    prediction_set=pred_set,
                    abstained=abstained,
                    alpha=self.alpha,
                    calibration_coverage=self.calibration_coverage or 0.0,
                    threshold=self.threshold,
                )
            )

        return results

    def get_coverage_report(self) -> dict:
        """Get calibration coverage statistics."""
        return {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": self.calibration_coverage or 0.0,
            "alpha": self.alpha,
            "n_calibration": len(self.calibration_scores)
            if self.calibration_scores is not None
            else 0,
            "threshold": self.threshold or 0.0,
            "score_type": self.score_type,
        }
