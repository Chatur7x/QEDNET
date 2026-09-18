"""Model interfaces and factory."""
from qednet.models.base import BaseEstimator, BaseTransformer
from qednet.models.factory import ModelFactory, create_model
from qednet.models.quantum import VQC, QSVM, ReuploadClassifier, HybridQNN
from qednet.models.quantum.hybrid import HybridQNNLayer, make_hybrid_model
from qednet.models.classical import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    XGBoostClassifier,
    LightGBMClassifier,
    MLPClassifier,
    RBFSVMModel,
    ParameterMatchedMLP,
)

__all__ = [
    "BaseEstimator", "BaseTransformer", "ModelFactory", "create_model",
    "VQC", "QSVM", "ReuploadClassifier", "HybridQNN",
    "HybridQNNLayer", "make_hybrid_model",
    "LogisticRegressionModel", "RandomForestModel", "XGBoostModel",
    "XGBoostClassifier", "LightGBMClassifier", "MLPClassifier",
    "RBFSVMModel", "ParameterMatchedMLP",
]