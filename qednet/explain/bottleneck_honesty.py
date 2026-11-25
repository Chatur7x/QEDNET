"""Bottleneck Honesty Meter (D9)."""

from __future__ import annotations

import numpy as np

from qednet.evaluation import nested_cv
from qednet.preprocess import PreprocessConfig, make_preprocess_pipeline


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

    def evaluate(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> dict:
        """Run bottleneck honesty comparison."""

        def preprocess_fn():
            return make_preprocess_pipeline(self.preprocess_config)

        # Both estimators receive the same fold-local compression. Fitting a
        # compressor once on the full dataset here would leak validation rows.
        qmodel = self.quantum_model_factory()
        q_result = nested_cv(
            qmodel,
            X,
            y,
            preprocess_fn,
            self.outer_folds,
            self.inner_folds,
            self.random_state,
            calibrate=groups is None,
            groups=groups,
        )

        # Classical comparator uses the same preprocessing and outer folds.
        cmodel = self.classical_model_factory()
        c_result = nested_cv(
            cmodel,
            X,
            y,
            preprocess_fn,
            self.outer_folds,
            self.inner_folds,
            self.random_state,
            calibrate=groups is None,
            groups=groups,
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
    groups: np.ndarray | None = None,
) -> dict:
    """Standalone function for bottleneck honesty check."""
    meter = BottleneckHonestyMeter(
        lambda: quantum_model,
        lambda: classical_model,
        preprocess_config,
        outer_folds,
        inner_folds,
        random_state,
    )
    return meter.evaluate(X, y, groups=groups)
