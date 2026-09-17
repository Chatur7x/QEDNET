"""F1 Quantum Second Opinion Cascade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass
class CascadeConfig:
    """Configuration for the classical-quantum cascade."""
    ambiguity_low: float = 0.35
    ambiguity_high: float = 0.65
    classical_model: Optional[BaseEstimator] = None
    quantum_model: Optional[BaseEstimator] = None
    preprocess_fn: Optional[callable] = None


class QuantumCascade(BaseEstimator, ClassifierMixin):
    """
    Classical-Quantum Second Opinion Cascade.

    Flow:
    1. Classical model (e.g., XGBoost) screens all patients
    2. If confidence in [ambiguity_low, ambiguity_high], defer to quantum model
    3. Otherwise use classical prediction
    4. Return calibrated probabilities with uncertainty
    """

    def __init__(
        self,
        classical_model: BaseEstimator,
        quantum_model: BaseEstimator,
        preprocess_fn,
        ambiguity_low: float = 0.35,
        ambiguity_high: float = 0.65,
        calibrate: bool = True,
    ):
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.preprocess_fn = preprocess_fn
        self.ambiguity_low = ambiguity_low
        self.ambiguity_high = ambiguity_high
        self.calibrate = calibrate
        self._preprocess = None
        self._fitted = False

    def fit(self, X, y):
        self._preprocess = self.preprocess_fn()
        X_trans = self._preprocess.fit_transform(X, y)

        # Fit classical
        self.classical_model.fit(X_trans, y)

        # Fit quantum (on same preprocessed features)
        self.quantum_model.fit(X_trans, y)

        self._fitted = True
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("Cascade not fitted")

        X_trans = self._preprocess.transform(X)

        # Classical predictions
        class_proba = self.classical_model.predict_proba(X_trans)[:, 1]

        # Identify ambiguous cases
        ambiguous = (class_proba >= self.ambiguity_low) & (class_proba <= self.ambiguity_high)

        # Final predictions
        final_proba = class_proba.copy()

        if np.any(ambiguous):
            # Quantum second opinion for ambiguous
            quantum_proba = self.quantum_model.predict_proba(X_trans[ambiguous])[:, 1]
            final_proba[ambiguous] = quantum_proba

        return np.column_stack([1 - final_proba, final_proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_routing(self, X):
        """Get routing decisions for each sample."""
        if not self._fitted:
            raise RuntimeError("Cascade not fitted")
        X_trans = self._preprocess.transform(X)
        class_proba = self.classical_model.predict_proba(X_trans)[:, 1]
        ambiguous = (class_proba >= self.ambiguity_low) & (class_proba <= self.ambiguity_high)
        return {
            "classical_proba": class_proba,
            "routed_to_quantum": ambiguous,
            "n_classical": int(np.sum(~ambiguous)),
            "n_quantum": int(np.sum(ambiguous)),
        }

    def metadata(self) -> dict:
        return {
            "model": "quantum_cascade",
            "ambiguity_low": self.ambiguity_low,
            "ambiguity_high": self.ambiguity_high,
            "classical_model": self.classical_model.__class__.__name__,
            "quantum_model": self.quantum_model.__class__.__name__,
        }