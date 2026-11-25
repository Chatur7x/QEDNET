"""Quantum Second-Opinion Cascade (F1)."""

from dataclasses import dataclass
from typing import Literal

from numpy.typing import NDArray

from qednet.models.base import BaseEstimator


@dataclass
class CascadeResult:
    """Result from cascade prediction."""

    prediction: int
    probability: float
    calibrated_probability: float
    routing: Literal["classical", "quantum_second_opinion"]
    reason: Literal["high_confidence", "ambiguous_probability"]
    classical_proba: float
    quantum_proba: float | None
    model_name: str
    config_hash: str


class QuantumSecondOpinionCascade:
    """Cascade that routes ambiguous classical predictions to quantum model.

    Architecture:
        Input
         ↓
        Classical screener (XGBoost)
         ↓
        Is probability ambiguous? (p ∈ [low, high])
         ↓ yes → Quantum model → Calibrated result
         ↓ no  → Use classical result
    """

    def __init__(
        self,
        classical_model: BaseEstimator,
        quantum_model: BaseEstimator,
        ambiguity_threshold: tuple[float, float] = (0.35, 0.65),
        classical_name: str = "xgboost",
        quantum_name: str = "vqc",
        config_hash: str = "",
    ):
        """Initialize cascade.

        Args:
            classical_model: Fitted classical screener
            quantum_model: Fitted quantum model
            ambiguity_threshold: (low, high) range for ambiguous probabilities.
                                Default (0.35, 0.65) means probabilities in [0.35, 0.65]
                                are considered ambiguous and routed to quantum.
            classical_name: Name of classical model for metadata
            quantum_name: Name of quantum model for metadata
            config_hash: Configuration hash for reproducibility
        """
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.ambiguity_low, self.ambiguity_high = ambiguity_threshold
        self.classical_name = classical_name
        self.quantum_name = quantum_name
        self.config_hash = config_hash

        if not (0 <= self.ambiguity_low < self.ambiguity_high <= 1):
            raise ValueError("ambiguity_threshold must satisfy 0 <= low < high <= 1")

    def predict(self, X: NDArray) -> list[CascadeResult]:
        """Run cascade prediction on batch of samples."""
        classical_probas = self.classical_model.predict_proba(X)[:, 1]
        quantum_probas = self.quantum_model.predict_proba(X)[:, 1]

        results = []
        for i in range(X.shape[0]):
            classical_p = float(classical_probas[i])
            quantum_p = float(quantum_probas[i])

            is_ambiguous = self.ambiguity_low <= classical_p <= self.ambiguity_high

            if is_ambiguous:
                calibrated_p = quantum_p
                routing = "quantum_second_opinion"
                reason = "ambiguous_probability"
                final_pred = int(quantum_p >= 0.5)
                model_name = self.quantum_name
            else:
                calibrated_p = classical_p
                routing = "classical"
                reason = "high_confidence"
                final_pred = int(classical_p >= 0.5)
                model_name = self.classical_name

            results.append(
                CascadeResult(
                    prediction=final_pred,
                    probability=calibrated_p,
                    calibrated_probability=calibrated_p,
                    routing=routing,
                    reason=reason,
                    classical_proba=classical_p,
                    quantum_proba=quantum_p if is_ambiguous else None,
                    model_name=model_name,
                    config_hash=self.config_hash,
                )
            )

        return results
