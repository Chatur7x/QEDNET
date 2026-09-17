"""Integration: ingestion -> validation -> preprocess -> model -> evaluation (F1-F7)."""

from __future__ import annotations

import numpy as np
import pytest

from qednet.data import DatasetRegistry
from qednet.evaluation.benchmark import make_preprocess_pipeline, nested_cv
from qednet.models import VQC
from qednet.models.classical import LogisticRegressionModel
from qednet.preprocess import PreprocessConfig


@pytest.fixture(scope="module")
def bundle():
    return DatasetRegistry.load("breast_cancer")


def test_ingestion_schema_ok(bundle):
    assert bundle.X.shape == (569, 30)
    assert set(np.unique(bundle.y)).issubset({0, 1})
    assert len(bundle.feature_names) == 30


def test_preprocess_leakage_safe(bundle):
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(
        bundle.X, bundle.y, test_size=0.2, random_state=0, stratify=bundle.y
    )
    pre = make_preprocess_pipeline(PreprocessConfig(select_k=8, compress_dim=4))
    Xtr_t = pre.fit_transform(Xtr, ytr)
    Xte_t = pre.transform(Xte)
    assert Xtr_t.shape == (455, 4) and Xte_t.shape == (114, 4)


def test_nested_cv_logistic(bundle):
    rng = np.random.RandomState(0)
    keep = np.concatenate([rng.choice(np.where(bundle.y == c)[0], 60, replace=False) for c in (0, 1)])
    X, y = bundle.X[keep], bundle.y[keep]
    pre_fn = lambda: make_preprocess_pipeline(PreprocessConfig(select_k=8, compress_dim=4))
    res = nested_cv(LogisticRegressionModel(), X, y, pre_fn, outer_folds=2,
                    random_state=0, calibrate=False, n_bootstrap=20)
    assert res["aggregated_metrics"]["auroc"]["mean"] > 0.9


def test_nested_cv_vqc_runs(bundle):
    rng = np.random.RandomState(1)
    keep = np.concatenate([rng.choice(np.where(bundle.y == c)[0], 25, replace=False) for c in (0, 1)])
    X, y = bundle.X[keep], bundle.y[keep]
    pre_fn = lambda: make_preprocess_pipeline(PreprocessConfig(select_k=6, compress_dim=4))
    res = nested_cv(VQC(n_qubits=4, n_layers=1, max_iter=2, seed=0), X, y, pre_fn,
                    outer_folds=2, random_state=1, calibrate=False, n_bootstrap=20)
    m = res["aggregated_metrics"]["auroc"]["mean"]
    assert 0.0 <= m <= 1.0
