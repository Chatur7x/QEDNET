"""QED-Net: A Hybrid Quantum–Classical ML Platform for Early Disease Detection.

This platform is a research and educational artifact. It is not a medical device
and must not be used for real patient care.

Member A: Quantum core (models/quantum, encoding, certificate, diagnostics, hardware)
Member B: Data, preprocessing, leakage guard
Member C: Platform & integration (train, evaluation, explain, deploy, API, dashboard)
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