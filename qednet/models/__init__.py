"""QED-Net Models - Quantum and Classical."""

from qednet.models.quantum import VQC, QSVM, ReuploadClassifier, HybridQNNLayer, make_hybrid_model
from qednet.models.classical import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    RBFSVMModel,
    ParameterMatchedMLP,
)

__all__ = [
    "VQC",
    "QSVM",
    "ReuploadClassifier",
    "HybridQNNLayer",
    "make_hybrid_model",
    "LogisticRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
    "RBFSVMModel",
    "ParameterMatchedMLP",
]