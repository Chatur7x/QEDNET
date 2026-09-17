"""ReuploadClassifier tests on tiny synthetic data."""

import numpy as np

from qednet.models.quantum.reupload import ReuploadClassifier


def _toy(n=12, d=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (n, d))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_init_and_params():
    m = ReuploadClassifier(n_qubits=2, n_layers=2)
    # theta: 2*2*3=12, phi: 2*2=4 => 16
    assert m.n_params == 16


def test_train_predict_shapes_ranges():
    X, y = _toy()
    m = ReuploadClassifier(n_qubits=2, n_layers=1, max_iter=5, seed=0).fit(X, y)
    P = m.predict_proba(X)
    assert P.shape == (len(X), 2)
    assert np.all(P >= 0) and np.all(P <= 1)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert m.predict(X).shape == (len(X),)
    assert len(m.train_loss_) == 5 and len(m.grad_variances_) == 5


def test_invalid_dims():
    X, y = _toy(d=3)
    import pytest
    with pytest.raises(ValueError):
        ReuploadClassifier(n_qubits=2).fit(X, y)  # 3 feats > 2 qubits


def test_deterministic_init():
    X, y = _toy()
    a = ReuploadClassifier(n_qubits=2, n_layers=1, max_iter=3, seed=1).fit(X, y)
    b = ReuploadClassifier(n_qubits=2, n_layers=1, max_iter=3, seed=1).fit(X, y)
    np.testing.assert_allclose(a.theta_, b.theta_)
    np.testing.assert_allclose(a.phi_, b.phi_)


def test_diff_methods():
    X, y = _toy(n=8)
    for dm in ["backprop", "parameter-shift"]:
        m = ReuploadClassifier(n_qubits=2, n_layers=1, max_iter=2, diff_method=dm).fit(X, y)
        assert m.predict_proba(X).shape == (8, 2)