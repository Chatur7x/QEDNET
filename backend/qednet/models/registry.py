"""Model registry: factory for classical and quantum models."""
from __future__ import annotations

from typing import Any, Dict, Type

from ..models.base import BaseModel
from .classical.baselines import (LogisticRegressionModel, MatchedMLPModel,
                                  RandomForestModel, RBFSVMModel,
                                  XGBoostModel)
from .quantum.hybrid_qnn import HybridQNN
from .quantum.qsvm import QSVM
from .quantum.reupload import DataReuploadingClassifier
from .quantum.vqc import VQC

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    # classical
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "rbf_svm": RBFSVMModel,
    "matched_mlp": MatchedMLPModel,
    # quantum
    "qsvm": QSVM,
    "vqc": VQC,
    "reupload": DataReuploadingClassifier,
    "hybrid_qnn": HybridQNN,
}

CLASSICAL_MODELS = ["logistic_regression", "random_forest", "xgboost",
                    "rbf_svm", "matched_mlp"]
QUANTUM_MODELS = ["qsvm", "vqc", "reupload", "hybrid_qnn"]


def build_model(model_key: str, **kwargs: Any) -> BaseModel:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_key}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_key](**kwargs)


def model_families() -> Dict[str, str]:
    return {k: v.family for k, v in MODEL_REGISTRY.items()}
