"""Serving pipeline combining model, cascade, conformal, and uncertainty."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from qednet.models.base import BaseEstimator
from qednet.deploy.cascade import QuantumSecondOpinionCascade
from qednet.deploy.conformal import ConformalPredictor, ConformalResult
from qednet.deploy.uncertainty import UncertaintyQuantifier, UncertaintyResult


@dataclass
class PredictionResult:
    """Complete prediction result from serving pipeline."""
    prediction: int
    probability: float
    calibrated_probability: float
    routing: str
    uncertainty: UncertaintyResult
    conformal: ConformalResult
    model_name: str
    config_hash: str
    abstained: bool = False


class ServingPipeline:
    """Production serving pipeline for QED-Net models."""

    def __init__(
        self,
        model: BaseEstimator,
        cascade: Optional[QuantumSecondOpinionCascade] = None,
        conformal: Optional[ConformalPredictor] = None,
        uncertainty: Optional[UncertaintyQuantifier] = None,
        model_name: str = "unknown",
        config_hash: str = "",
    ):
        self.model = model
        self.cascade = cascade
        self.conformal = conformal
        self.uncertainty = uncertainty
        self.model_name = model_name
        self.config_hash = config_hash

    def predict(self, X: NDArray) -> list[PredictionResult]:
        """Run prediction through full serving pipeline."""
        results = []

        for i in range(X.shape[0]):
            x = X[i:i+1]

            # 1. Get base prediction
            proba = self.model.predict_proba(x)[0]
            prob_class_1 = proba[1] if len(proba) > 1 else proba[0]
            pred = int(prob_class_1 >= 0.5)

            # 2. Apply cascade if available
            routing = "direct"
            calibrated_prob = prob_class_1

            if self.cascade:
                cascade_result = self.cascade.predict(x)
                pred = cascade_result[0].prediction
                calibrated_prob = cascade_result[0].calibrated_probability
                routing = cascade_result[0].routing

            # 3. Quantify uncertainty
            uncertainty_result = None
            if self.uncertainty:
                uncertainty_result = self.uncertainty.quantify(calibrated_prob)

            # 4. Conformal prediction
            conformal_result = None
            abstained = False
            if self.conformal:
                conformal_result = self.conformal.predict(x)[0]
                abstained = conformal_result.abstained

            results.append(PredictionResult(
                prediction=pred,
                probability=prob_class_1,
                calibrated_probability=calibrated_prob,
                routing=routing,
                uncertainty=uncertainty_result,
                conformal=conformal_result,
                model_name=self.model_name,
                config_hash=self.config_hash,
                abstained=abstained,
            ))

        return results

    def predict_single(self, x: NDArray) -> PredictionResult:
        """Predict single sample."""
        return self.predict(x.reshape(1, -1))[0]