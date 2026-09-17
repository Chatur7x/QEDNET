"""QSVM tests: kernel properties, fit/predict, dim validation."""

import numpy as np

from qednet.models.quantum.qsvm import QSVM


def _toy(n=10, d=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (n, d))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_init_and_metadata():
    m = QSVM(n_qubits=2, encoding="zz")
    assert m.n_params == 0
    meta = m.metadata()
    assert meta["model"] == "qsvm"
    assert meta["n_qubits"] == 2


def test_kernel_symmetry_and_diagonal():
    X, _ = _toy(n=6)
    m = QSVM(n_qubits=2, encoding="zz").fit(X, (X[:, 0] > 0).astype(int))
    # test via internal kernel compute
    K = m._compute_kernel(X)
    assert K.shape == (6, 6)
    # symmetry
    np.testing.assert_allclose(K, K.T, atol=1e-6)
    # diagonal == 1 (|<x|x>|^2 = 1)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)


def test_fit_predict_shapes():
    X, y = _toy()
    m = QSVM(n_qubits=2, encoding="zz").fit(X, y)
    P = m.predict_proba(X)
    assert P.shape == (len(X), 2)
    assert np.all(P >= 0) and np.all(P <= 1)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    preds = m.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_invalid_dims():
    X, y = _toy(d=3)
    import pytest
    with pytest.raises(ValueError):
        QSVM(n_qubits=2).fit(X, y)


def test_swap_test_mode():
    X, y = _toy(n=8)
    m = QSVM(n_qubits=2, encoding="zz", use_swap_test=True).fit(X, y)
    P = m.predict_proba(X)
    assert P.shape == (8, 2)


def test_kernel_evals_exposed():
    X, y = _toy(n=5)
    m = QSVM(n_qubits=2).fit(X, y)
    # n(n+1)/2 for symmetric kernel computation
    assert m.kernel_evals_ >= 15
    meta = m.metadata()
    assert "kernel_evals" in meta