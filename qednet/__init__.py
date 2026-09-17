"""QED-Net: A Hybrid Quantum–Classical ML Platform for Early Disease Detection.

This platform is a research and educational artifact. It is not a medical device
and must not be used for real patient care.
"""

__version__ = "0.1.0"
__author__ = "QED-Net Team"

from qednet.models.base import BaseEstimator, BaseTransformer
from qednet.train.config import ExperimentConfig
from qednet.train.runner import TrainingRunner, run_experiment

__all__ = [
    "BaseEstimator",
    "BaseTransformer",
    "ExperimentConfig",
    "TrainingRunner",
    "run_experiment",
]