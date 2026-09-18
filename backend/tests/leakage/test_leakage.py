"""Leakage tests (F2) — deliberate leakage attempts MUST be detected.

If any of these tests fail in CI: TEST FAILURE / EXPERIMENT INVALID.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qednet.data.validation import (LeakageError, LeakageGuard,
                                    check_train_test_contamination,
                                    validate_frame)
from qednet.preprocess.pipeline import PreprocessPipeline


def _make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_scaler_fit_on_test_raises():
    """Fitting the scaler on test data must raise LeakageError."""
    X, y = _make_data()
    guard = LeakageGuard()
    pipe = PreprocessPipeline(target_dim=4, guard=guard)
    guard.allow_fit()
    pipe.fit(X[:150], y[:150])
    guard.forbid_fit("test")
    with pytest.raises(LeakageError):
        pipe.imputer.fit(X[150:])  # any learned refit now forbidden


def test_pipeline_transform_never_fits():
    """Transform on unseen data must not alter fitted statistics."""
    X, y = _make_data()
    pipe = PreprocessPipeline(target_dim=4)
    pipe.fit(X[:150], y[:150])
    mu_before = pipe.scaler.mean_.copy()
    pipe.transform(X[150:] * 5.0)  # extreme shift
    assert np.allclose(pipe.scaler.mean_, mu_before)


def test_no_train_test_row_contamination():
    X, _ = _make_data(n=100, seed=3)
    tr, te = X[:80], X[80:]
    rep = check_train_test_contamination(tr, te)
    assert rep["contaminated"] is False

    rep_bad = check_train_test_contamination(tr, np.vstack([tr[:5], te]))
    assert rep_bad["contaminated"] is True
    assert rep_bad["n_shared_rows"] == 5


def test_pca_fit_on_train_only():
    """PCA must be fit on training data only; test transform must not
    change components."""
    X, y = _make_data()
    pipe = PreprocessPipeline(target_dim=3)
    pipe.fit(X[:150], y[:150])
    comps = pipe.compressor.components_.copy()
    pipe.transform(X[150:])
    assert np.allclose(pipe.compressor.components_, comps)


def test_smote_only_on_training_folds():
    """SMOTE resampling is exposed only as resample_train (train folds)."""
    X, y = _make_data(120)
    pipe = PreprocessPipeline(target_dim=4, handle_imbalance=True)
    pipe.fit(X[:90], y[:90])
    Xr, yr = pipe.resample_train(pipe.transform(X[:90]), y[:90])
    assert len(yr) >= 90  # minority oversampled (or unchanged)
    # transform on test must still work and not resample
    Zt = pipe.transform(X[90:])
    assert Zt.shape[0] == 30


def test_validation_catches_bad_target():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 0, 1], "t": [0, 1, 0]})
    rep = validate_frame(df, "missing_target")
    assert rep.passed is False
    assert rep.checks["target_present"] is False


def test_validation_catches_non_binary_target():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "t": [0, 1, 2, 3]})
    rep = validate_frame(df, "t")
    assert rep.passed is False


def test_validation_flags_identifier_columns():
    df = pd.DataFrame({"patient_name": ["a", "b", "c", "d"],
                       "f1": [1.0, 2.0, 3.0, 4.0],
                       "t": [0, 1, 0, 1]})
    rep = validate_frame(df, "t")
    assert "patient_name" in rep.stats["identifier_like_columns"]
    assert any("identifier" in w or "Direct-identifier" in w
               for w in rep.warnings)


def test_guard_audit_reports_all_parts():
    X, y = _make_data()
    guard = LeakageGuard()
    pipe = PreprocessPipeline(target_dim=4, guard=guard)
    guard.allow_fit()
    pipe.fit(X, y)
    guard.forbid_fit("eval")
    pipe.transform(X)
    audit = guard.audit()
    assert set(audit["fitted_parts"]) >= {"imputer", "scaler", "pca"}
    assert audit["contamination_detected"] is False
