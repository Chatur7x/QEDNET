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
    keep = np.concatenate(
        [rng.choice(np.where(bundle.y == c)[0], 60, replace=False) for c in (0, 1)]
    )
    X, y = bundle.X[keep], bundle.y[keep]

    def pre_fn():
        return make_preprocess_pipeline(PreprocessConfig(select_k=8, compress_dim=4))

    res = nested_cv(
        LogisticRegressionModel(),
        X,
        y,
        pre_fn,
        outer_folds=2,
        random_state=0,
        calibrate=False,
        n_bootstrap=20,
    )
    assert res["aggregated_metrics"]["auroc"]["mean"] > 0.9


def test_nested_cv_keeps_groups_out_of_other_folds():
    from sklearn.base import BaseEstimator, TransformerMixin

    class GroupCheckingPreprocessor(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.train_groups_ = set(X[:, 0].astype(int))
            return self

        def fit_transform(self, X, y=None):
            self.fit(X, y)
            return X[:, 1:]

        def transform(self, X):
            test_groups = set(X[:, 0].astype(int))
            assert self.train_groups_.isdisjoint(test_groups)
            return X[:, 1:]

    groups = np.repeat(np.arange(12), 4)
    y = np.tile([0, 1, 0, 1], 12)
    X = np.column_stack([groups, np.random.RandomState(3).normal(size=len(groups))])
    result = nested_cv(
        LogisticRegressionModel(),
        X,
        y,
        GroupCheckingPreprocessor,
        outer_folds=3,
        random_state=42,
        calibrate=False,
        n_bootstrap=10,
        groups=groups,
    )
    assert len(result["fold_results"]) == 3


def test_grouped_cv_rejects_non_group_aware_calibration():
    groups = np.repeat(np.arange(6), 4)
    y = np.tile([0, 1, 0, 1], 6)
    X = np.random.RandomState(4).normal(size=(len(y), 3))
    with pytest.raises(ValueError, match="group-aware"):
        nested_cv(
            LogisticRegressionModel(),
            X,
            y,
            lambda: make_preprocess_pipeline(PreprocessConfig(compress_dim=2)),
            outer_folds=3,
            random_state=42,
            groups=groups,
        )


def test_nested_cv_vqc_runs(bundle):
    rng = np.random.RandomState(1)
    keep = np.concatenate(
        [rng.choice(np.where(bundle.y == c)[0], 25, replace=False) for c in (0, 1)]
    )
    X, y = bundle.X[keep], bundle.y[keep]

    def pre_fn():
        return make_preprocess_pipeline(PreprocessConfig(select_k=6, compress_dim=4))

    res = nested_cv(
        VQC(n_qubits=4, n_layers=1, max_iter=2, seed=0),
        X,
        y,
        pre_fn,
        outer_folds=2,
        random_state=1,
        calibrate=False,
        n_bootstrap=20,
    )
    m = res["aggregated_metrics"]["auroc"]["mean"]
    assert 0.0 <= m <= 1.0
