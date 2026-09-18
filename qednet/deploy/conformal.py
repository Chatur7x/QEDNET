"""Conformal Prediction (F6)."""

from dataclasses import dataclass
from typing import List
import numpy as np
from numpy.typing import NDArray

from qednet.models.base import BaseEstimator


@dataclass
class ConformalResult:
    """Result from conformal prediction."""
    prediction_set: List[int]
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
    ):
        """Initialize conformal predictor.

        Args:
            model: Fitted base model with predict_proba method
            alpha: Miscoverage level (1 - alpha = target coverage)
            score_type: Nonconformity score type ('max_proba', 'margin', 'aps')
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

        self.model = model
        self.alpha = alpha
        self.score_type = score_type
        self.calibration_scores: NDArray | None = None
        self.threshold: float | None = None
        self.calibration_coverage: float | None = None
        self._is_calibrated = False

    def _compute_scores(self, y_true: NDArray, y_proba: NDArray) -> NDArray:
        """Compute nonconformity scores."""
        proba_1 = y_proba[:, 1] if y_proba.ndim > 1 else y_proba

        if self.score_type == "max_proba":
            true_class_proba = y_proba[np.arange(len(y_true)), y_true]
            return 1 - true_class_proba
        elif self.score_type == "margin":
            margin = np.abs(proba_1 - (1 - proba_1))
            return 1 - margin
        elif self.score_type == "aps":
            # Adaptive Prediction Sets (Romano et al., 2020)
            sorted_indices = np.argsort(y_proba, axis=1)[:, ::-1]
            ranks = np.argsort(sorted_indices, axis=1)
            true_ranks = ranks[np.arange(len(y_true)), y_true]
            scores = np.zeros(len(y_true))
            for i in range(len(y_true)):
                higher = sorted_indices[i, :true_ranks[i]]
                scores[i] = y_proba[i, higher].sum() if len(higher) > 0 else 0
            scores += np.random.rand(len(y_true)) * y_proba[np.arange(len(y_true)), y_true]
            return scores
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

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

        y_proba = self.model.predict_proba(X_cal)
        self.calibration_scores = self._compute_scores(y_cal, y_proba)

        n_cal = len(self.calibration_scores)
        quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        quantile_level = min(quantile_level, 1.0)
        self.threshold = float(np.quantile(self.calibration_scores, quantile_level))
        self._is_calibrated = True

        pred_sets = self._get_prediction_sets(y_proba)
        coverage = np.mean([y_cal[i] in pred_sets[i] for i in range(len(y_cal))])
        self.calibration_coverage = float(coverage)

        return self

    def _get_prediction_sets(self, y_proba: NDArray) -> List[List[int]]:
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
                margin = abs(proba[0] - proba[1])
                score_0 = 1 - margin
                score_1 = score_0
            else:  # aps
                sorted_idx = np.argsort(proba)[::-1]
                rank_0 = np.where(sorted_idx == 0)[0][0]
                higher_0 = sorted_idx[:rank_0]
                score_0 = proba[higher_0].sum() if len(higher_0) > 0 else 0

                rank_1 = np.where(sorted_idx == 1)[0][0]
                higher_1 = sorted_idx[:rank_1]
                score_1 = proba[higher_1].sum() if len(higher_1) > 0 else 0

            scores = [score_0, score_1]
            pred_set = [c for c, s in enumerate(scores) if s <= self.threshold]
            pred_sets.append(pred_set)

        return pred_sets

    def predict(self, X: NDArray) -> List[ConformalResult]:
        """Generate conformal prediction sets."""
        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        y_proba = self.model.predict_proba(X)
        pred_sets = self._get_prediction_sets(y_proba)

        results = []
        for pred_set in pred_sets:
            abstained = len(pred_set) == 2
            results.append(ConformalResult(
                prediction_set=pred_set,
                abstained=abstained,
                alpha=self.alpha,
                calibration_coverage=self.calibration_coverage or 0.0,
                threshold=self.threshold,
            ))

        return results

    def get_coverage_report(self) -> dict:
        """Get calibration coverage statistics."""
        return {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": self.calibration_coverage or 0.0,
            "alpha": self.alpha,
            "n_calibration": len(self.calibration_scores) if self.calibration_scores is not None else 0,
            "threshold": self.threshold or 0.0,
            "score_type": self.score_type,
        }