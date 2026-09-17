"""Shot-noise uncertainty quantification (N5)."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class UncertaintyResult:
    """Uncertainty quantification result."""
    probability: float
    shots: int
    variance: float
    std: float
    ci_lower: float
    ci_upper: float
    # Epistemic uncertainty (if ensemble available)
    ensemble_variance: float | None = None
    ensemble_std: float | None = None


class UncertaintyQuantifier:
    """Quantify uncertainty from quantum shot noise.

    For estimated probability p and S shots:
        Var(p_hat) = p(1-p) / S

    IMPORTANT: This is shot/sampling uncertainty, NOT model uncertainty.
    Model uncertainty (epistemic) requires ensemble methods.
    """

    def __init__(self, shots: int = 1024, confidence: float = 0.95):
        """Initialize uncertainty quantifier.

        Args:
            shots: Number of quantum shots
            confidence: Confidence level for intervals
        """
        self.shots = shots
        self.confidence = confidence
        self.z_score = self._get_z_score(confidence)

    def _get_z_score(self, confidence: float) -> float:
        from scipy.stats import norm
        alpha = (1 - confidence) / 2
        return norm.ppf(1 - alpha)

    def quantify(self, probability: float, ensemble_probs: NDArray | None = None) -> UncertaintyResult:
        """Quantify uncertainty for a probability estimate.

        Args:
            probability: Estimated probability (0-1)
            ensemble_probs: Optional array of ensemble predictions for epistemic uncertainty

        Returns:
            UncertaintyResult with variance, std, and confidence intervals
        """
        p = np.clip(probability, 1e-10, 1 - 1e-10)
        var = p * (1 - p) / self.shots
        std = np.sqrt(var)

        ci_lower = max(0, p - self.z_score * std)
        ci_upper = min(1, p + self.z_score * std)

        result = UncertaintyResult(
            probability=probability,
            shots=self.shots,
            variance=var,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

        # Add epistemic uncertainty if ensemble provided
        if ensemble_probs is not None and len(ensemble_probs) > 1:
            ensemble_var = np.var(ensemble_probs)
            ensemble_std = np.sqrt(ensemble_var)
            result.ensemble_variance = ensemble_var
            result.ensemble_std = ensemble_std

        return result

    def quantify_batch(self, probabilities: NDArray, ensemble_probs: NDArray | None = None) -> list[UncertaintyResult]:
        """Quantify uncertainty for batch of probabilities."""
        return [self.quantify(p, ensemble_probs[i] if ensemble_probs is not None else None)
                for i, p in enumerate(probabilities)]