"""Bottleneck Honesty Meter (D9)."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from qednet.evaluation import nested_cv
from qednet.preprocess import make_preprocess_pipeline, PreprocessConfig


class BottleneckHonestyMeter:
    """
    Compare quantum model against a classical model trained on the same
    compressed features to estimate quantum contribution after compression.
    """

    def __init__(
        self,
        quantum_model_factory,
        classical_model_factory,
        preprocess_config: PreprocessConfig,
        outer_folds: int = 5,
        inner_folds: int = 3,
        random_state: int = 42,
    ):
        self.quantum_model_factory = quantum_model_factory
        self.classical_model_factory = classical_model_factory
        self.preprocess_config = preprocess_config
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.random_state = random_state

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run bottleneck honesty comparison."""
        preprocess_fn = lambda: make_preprocess_pipeline(self.preprocess_config)

        # Get compressed features
        preprocess = make_preprocess_pipeline(self.preprocess_config)
        X_compressed = preprocess.fit_transform(X, y)

        # Quantum model on compressed features
        qmodel = self.quantum_model_factory()
        q_result = nested_cv(
            qmodel, X, y, preprocess_fn,
            self.outer_folds, self.inner_folds, self.random_state
        )

        # Classical model on same compressed features
        class preprocess_for_classical:
            def __init__(self, compressor):
                self.compressor = compressor
            def fit_transform(self, X, y):
                return self.compressor.transform(X)
            def transform(self, X):
                return self.compressor.transform(X)

        classical_preprocess = preprocess_for_classical(preprocess)
        cmodel = self.classical_model_factory()
        c_result = nested_cv(
            cmodel, X_compressed, y,
            lambda: classical_preprocess,
            self.outer_folds, self.inner_folds, self.random_state
        )

        # Compare
        q_metrics = q_result["aggregated_metrics"]
        c_metrics = c_result["aggregated_metrics"]

        comparison = {}
        for metric in q_metrics:
            qm = q_metrics[metric]["mean"]
            cm = c_metrics[metric]["mean"]
            comparison[metric] = {
                "quantum": qm,
                "classical_on_compressed": cm,
                "difference": qm - cm,
                "relative_gain": (qm - cm) / cm if cm != 0 else 0,
            }

        return {
            "quantum_metrics": q_metrics,
            "classical_on_compressed_metrics": c_metrics,
            "comparison": comparison,
            "interpretation": self._interpret(comparison),
        }

    def _interpret(self, comparison: dict) -> str:
        auroc_diff = comparison.get("auroc", {}).get("difference", 0)
        if auroc_diff > 0.02:
            return "Quantum model shows meaningful advantage over classical on compressed features"
        elif auroc_diff > -0.02:
            return "Quantum and classical models perform similarly on compressed features"
        else:
            return "Classical model outperforms quantum on compressed features"


def bottleneck_honesty_check(
    quantum_model,
    classical_model,
    X: np.ndarray,
    y: np.ndarray,
    preprocess_config: PreprocessConfig,
    outer_folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 42,
) -> dict:
    """Standalone function for bottleneck honesty check."""
    meter = BottleneckHonestyMeter(
        lambda: quantum_model,
        lambda: classical_model,
        preprocess_config,
        outer_folds, inner_folds, random_state
    )
    return meter.evaluate(X, y)