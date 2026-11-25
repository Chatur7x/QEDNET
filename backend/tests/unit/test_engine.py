"""Evaluation-engine integrity tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qednet.config import ExperimentConfig
from qednet.evaluation import engine as engine_module
from qednet.evaluation.engine import BenchmarkEngine, INNER_GRIDS
from qednet.models.quantum.hybrid_qnn import HybridQNN
from qednet.models.quantum.reupload import DataReuploadingClassifier
from qednet.models.quantum.vqc import VQC
from qednet.train.runner import _analysis_split_indices


def test_inner_tuning_fits_preprocessing_on_inner_train_only(monkeypatch):
    rng = np.random.default_rng(84)
    X = rng.normal(size=(90, 6))
    y = np.tile([0, 1], 45)
    fit_sizes: list[int] = []
    original_pipeline = engine_module.PreprocessPipeline

    class RecordingPipeline(original_pipeline):
        def fit(self, X_fit, y_fit, feature_names=None):
            fit_sizes.append(len(X_fit))
            return super().fit(X_fit, y_fit, feature_names)

    monkeypatch.setattr(engine_module, "PreprocessPipeline", RecordingPipeline)
    cfg = ExperimentConfig()
    cfg.compression.target_dim = 3
    cfg.training.inner_folds = 3
    benchmark = BenchmarkEngine(cfg, tuning_enabled=True)

    best = benchmark._inner_tune(
        "logistic_regression", {}, X, y, seed=17,
        feature_names=[f"feature_{i}" for i in range(X.shape[1])],
    )

    expected_fits = len(INNER_GRIDS["logistic_regression"]["C"])
    expected_fits *= cfg.training.inner_folds
    assert len(fit_sizes) == expected_fits
    assert all(size < len(X) for size in fit_sizes)
    assert best["C"] in INNER_GRIDS["logistic_regression"]["C"]


def test_grouped_folds_keep_subjects_disjoint():
    y = np.tile([0, 1], 24)
    groups = np.repeat(np.arange(24), 2)
    split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, test_idx in split.split(np.zeros((len(y), 1)), y, groups):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_calibrator_uses_cross_fitted_model_probabilities():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 6))
    y = (X[:, 0] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)
    cfg = ExperimentConfig()
    cfg.compression.target_dim = 3
    cfg.training.inner_folds = 3
    cfg.evaluation.calibration = "platt"
    benchmark = BenchmarkEngine(cfg, tuning_enabled=False)

    calibrator = benchmark._crossfit_calibrator(
        "logistic_regression", X, y, seed=42,
        feature_names=[f"feature_{i}" for i in range(X.shape[1])],
        groups=None, matched_params=None,
    )

    calibrated = calibrator.transform(np.array([0.2, 0.8]))
    assert calibrated.shape == (2,)
    assert np.isfinite(calibrated).all()
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_unimplemented_pennylane_backends_fail_clearly():
    for model_class in (VQC, DataReuploadingClassifier, HybridQNN):
        try:
            model_class(backend="pennylane")
        except ValueError as exc:
            assert "not implemented" in str(exc)
        else:
            raise AssertionError(
                f"{model_class.__name__} accepted an unavailable backend")


def test_certificate_auxiliary_split_keeps_subjects_disjoint():
    groups = np.repeat(np.arange(24), 2)
    y = np.repeat(np.tile([0, 1], 12), 2)
    train_idx, test_idx = _analysis_split_indices(y, seed=42, groups=groups)
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
