"""Leakage guard tests (F2). Deliberate leakage MUST be caught — failure = invalid experiment."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qednet.data import DatasetRegistry
from qednet.evaluation.benchmark import make_preprocess_pipeline
from qednet.evaluation.leakage_guard import (
    audit_pipeline,
    check_preprocessing_leakage,
    detect_leakage,
)
from qednet.preprocess import PreprocessConfig


@pytest.fixture(scope="module")
def data():
    b = DatasetRegistry.load("breast_cancer")
    return b.X, b.y


def test_preprocessing_guard_fires_on_full_fit(data):
    """Scaler/PCA fit on full data differs from fold-fit → guard must flag it."""
    X, y = data
    pre_fn = lambda: make_preprocess_pipeline(PreprocessConfig(select_k=8, compress_dim=4))
    rep = check_preprocessing_leakage(pre_fn, X, y, cv=3)
    assert rep["leakage_detected"], "guard failed to detect full-data preprocessing fit"


def test_target_leak_feature_is_caught(data):
    """Appending y as a feature must be flagged suspicious."""
    X, y = data
    X_leak = np.hstack([X, y.reshape(-1, 1)])
    rep = detect_leakage(LogisticRegression(max_iter=500), X_leak, y, cv=3)
    assert rep["suspicious"] or len(rep["high_corr_features"]) > 0


def test_clean_data_not_flagged(data):
    X, y = data
    # easy dataset scores high honestly; only a literally-perfect score is suspicious
    rep = detect_leakage(LogisticRegression(max_iter=500), X, y, cv=3, threshold=1.0)
    assert not rep["suspicious"]
    assert rep["cv_auc_mean"] > 0.9  # sanity: real signal, not broken data
    assert len(rep["high_corr_features"]) == 0


def test_audit_pipeline_runs(data):
    X, y = data
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    rep = audit_pipeline(pipe, X, y)
    assert isinstance(rep, dict)
