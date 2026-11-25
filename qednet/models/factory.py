"""Model factory for creating models from configuration."""

from typing import Any

from qednet.models.base import BaseEstimator
from qednet.models.classical.lightgbm_model import LightGBMClassifier
from qednet.models.classical.logistic_regression import LogisticRegressionModel
from qednet.models.classical.mlp import MLPClassifier
from qednet.models.classical.xgboost_model import XGBoostClassifier
from qednet.models.quantum.hybrid_qnn import HybridQNN
from qednet.models.quantum.qsvm import QSVM
from qednet.models.quantum.reupload import ReuploadClassifier
from qednet.models.quantum.vqc import VQC


class ModelFactory:
    """Factory for creating model instances from configuration."""

    _QUANTUM_MODELS = {
        "vqc": VQC,
        "qsvm": QSVM,
        "reupload": ReuploadClassifier,
        "hybrid_qnn": HybridQNN,
    }

    _CLASSICAL_MODELS = {
        "xgboost": XGBoostClassifier,
        "lightgbm": LightGBMClassifier,
        "logistic": LogisticRegressionModel,
        "mlp": MLPClassifier,
    }

    @classmethod
    def create_model(cls, model_config: Any, use_mock: bool = False) -> BaseEstimator:
        """Create a model instance from configuration.

        Args:
            model_config: ModelConfig object or dict with model parameters
            use_mock: If True, use mock implementations for quantum models

        Returns:
            Model instance implementing BaseEstimator
        """
        if hasattr(model_config, "type"):
            model_type = model_config.type
            params = {k: v for k, v in model_config.__dict__.items() if k != "type"}
        else:
            model_type = model_config.get("type", "vqc")
            params = {k: v for k, v in model_config.items() if k != "type"}

        # Handle quantum models
        if model_type in cls._QUANTUM_MODELS:
            if use_mock:
                from qednet.models.quantum.mocks import get_mock_model

                return get_mock_model(model_type, **params)
            return cls._QUANTUM_MODELS[model_type](**params)

        # Handle classical models
        if model_type in cls._CLASSICAL_MODELS:
            return cls._CLASSICAL_MODELS[model_type](**params)

        raise ValueError(f"Unknown model type: {model_type}")

    @classmethod
    def list_quantum_models(cls) -> list[str]:
        return list(cls._QUANTUM_MODELS.keys())

    @classmethod
    def list_classical_models(cls) -> list[str]:
        return list(cls._CLASSICAL_MODELS.keys())


def create_model(model_config: Any, use_mock: bool = False) -> BaseEstimator:
    """Convenience function to create a model."""
    return ModelFactory.create_model(model_config, use_mock)
