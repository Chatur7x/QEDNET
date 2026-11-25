"""QED-Net: A research platform for hybrid quantum-classical ML experiments.

This platform is a research and educational artifact. It is not a medical device
and must not be used for real patient care.

Member A: Quantum core (models/quantum, encoding, certificate, diagnostics, hardware)
Member B: Data, preprocessing, leakage guard
Member C: Platform & integration (train, evaluation, explain, deploy, API, dashboard)
"""

__version__ = "0.1.0"
__author__ = "QED-Net Team"

__all__ = [
    "BaseEstimator",
    "BaseTransformer",
    "ExperimentConfig",
    "TrainingRunner",
    "run_experiment",
]


def __getattr__(name: str):
    """Load the public API lazily so lightweight imports stay lightweight."""
    if name in {"BaseEstimator", "BaseTransformer"}:
        from qednet.models.base import BaseEstimator, BaseTransformer

        return {"BaseEstimator": BaseEstimator, "BaseTransformer": BaseTransformer}[name]
    if name == "ExperimentConfig":
        from qednet.train.config import ExperimentConfig

        return ExperimentConfig
    if name in {"TrainingRunner", "run_experiment"}:
        from qednet.train.runner import TrainingRunner, run_experiment

        return {"TrainingRunner": TrainingRunner, "run_experiment": run_experiment}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
