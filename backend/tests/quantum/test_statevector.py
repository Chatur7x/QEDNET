"""Quantum tests: simulator vs PennyLane equivalence + circuit properties.

The batched NumPy statevector simulator MUST reproduce PennyLane
default.qubit to machine precision on identical circuits. QSVM kernels must
be PSD-valid fidelity kernels from real feature-map states.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

qt = pytest.importorskip("pennylane")

from qednet.sim.statevector import (BatchedCircuit, ladder_entanglers,
                                    ring_entanglers)


def _reference_circuit(n, X, W, layers, entangle=True):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(x, w):
        for q in range(n):
            qml.RY(np.pi * x[q], wires=q)
        for l in range(layers):
            for q in range(n):
                qml.RY(w[l, q], wires=q)
            if entangle:
                for c, t in ladder_entanglers(n):
                    qml.CNOT(wires=[c, t])
        return qml.expval(qml.PauliZ(0))

    return circuit


def test_statevector_matches_pennylane_expval():
    n, B, layers = 5, 8, 3
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, (B, n))
    W = rng.uniform(0, 2 * np.pi, (layers, n))

    cir = BatchedCircuit(n)
    state = cir.zero_state(B)
    for q in range(n):
        state = cir.ry(state, q, np.pi * X[:, q])
    for l in range(layers):
        for q in range(n):
            state = cir.ry(state, q, W[l, q])
        for c, t in ladder_entanglers(n):
            state = cir.cnot(state, c, t)
    mine = np.asarray(cir.expval_z(state, 0))

    ref = np.array([_reference_circuit(n, None, W, layers)(X[b], W)
                    for b in range(B)])
    assert np.allclose(mine, ref, atol=1e-12)


def test_statevector_matches_pennylane_statevector():
    n, B = 4, 5
    rng = np.random.default_rng(11)
    X = rng.uniform(0, 1, (B, n))
    import pennylane as qml
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def state_circuit(x):
        for q in range(n):
            qml.RY(np.pi * x[q], wires=q)
        for c, t in ring_entanglers(n):
            qml.CNOT(wires=[c, t])
        return qml.state()

    cir = BatchedCircuit(n)
    state = cir.zero_state(B)
    for q in range(n):
        state = cir.ry(state, q, np.pi * X[:, q])
    for c, t in ring_entanglers(n):
        state = cir.cnot(state, c, t)
    mine = np.asarray(state).reshape(B, -1)
    ref = np.array([state_circuit(x) for x in X])
    assert np.allclose(mine, ref, atol=1e-12)


def test_cnot_permutation_is_involutory():
    from qednet.sim.statevector import _cnot_permutation
    for n in (3, 4, 5):
        for c in range(n):
            for t in range(n):
                if c == t:
                    continue
                perm = _cnot_permutation(n, c, t)
                assert np.array_equal(perm[perm], np.arange(2 ** n))


def test_expval_z_bounds_and_norm():
    n, B = 4, 16
    rng = np.random.default_rng(5)
    cir = BatchedCircuit(n)
    state = cir.zero_state(B)
    for q in range(n):
        state = cir.ry(state, q, rng.uniform(0, np.pi, B))
    out = np.asarray(cir.expval_z(state, 2))
    assert np.all(out >= -1 - 1e-12) and np.all(out <= 1 + 1e-12)
    # |0...0> single-qubit RY theta on qubit 2
    cir2 = BatchedCircuit(n)
    s2 = cir2.zero_state(1)
    theta = 0.7
    s2 = cir2.ry(s2, 2, np.array([theta]))
    assert abs(float(np.asarray(cir2.expval_z(s2, 2))[0]) -
               np.cos(theta)) < 1e-12


def test_zz_feature_map_states_unit_norm_and_symmetric_kernel():
    from qednet.encoding.zz import ZZFeatureMap
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (12, 5))
    enc = ZZFeatureMap(n_qubits=5, reps=2)
    states = enc.states(X)
    norms = np.linalg.norm(states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)
    K = np.abs(states.conj() @ states.T) ** 2
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.allclose(np.diag(K), 1.0, atol=1e-10)
    eig = np.linalg.eigvalsh(K)
    assert eig.min() > -1e-10  # PSD fidelity kernel


def test_angle_encoder_states_match_pennylane():
    from qednet.encoding.angle import AngleEncoder
    import pennylane as qml
    rng = np.random.default_rng(9)
    X = rng.uniform(0, 1, (6, 4))
    enc = AngleEncoder(n_qubits=4)
    mine = enc.states(X)
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def st(x):
        for q in range(4):
            qml.RY(np.pi * x[q], wires=q)
        return qml.state()

    ref = np.array([st(x) for x in X])
    assert np.allclose(mine, ref, atol=1e-12)


def test_zz_encoder_matches_pennylane():
    from qednet.encoding.zz import ZZFeatureMap
    import pennylane as qml
    rng = np.random.default_rng(13)
    X = rng.uniform(0, 1, (5, 4))
    enc = ZZFeatureMap(n_qubits=4, reps=2)
    mine = enc.states(X)
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def st(x):
        for q in range(4):
            qml.Hadamard(wires=q)
        for _ in range(2):
            for q in range(4):
                qml.RZ(2.0 * x[q], wires=q)
            for c in range(4):
                t = (c + 1) % 4
                qml.CNOT(wires=[c, t])
                qml.RZ(2.0 * x[c] * x[t], wires=t)
                qml.CNOT(wires=[c, t])
        return qml.state()

    ref = np.array([st(x) for x in X])
    assert np.allclose(mine, ref, atol=1e-10)


def test_amplitude_encoder_validation_and_normalisation():
    from qednet.encoding.amplitude import AmplitudeEncoder
    from qednet.encoding.base import EncodingError
    enc = AmplitudeEncoder(n_qubits=3)
    X = np.random.default_rng(0).normal(size=(4, 6))
    out = enc.encode(X)
    assert out.states.shape == (4, 8)
    assert np.allclose(np.linalg.norm(out.states, axis=1), 1.0)
    with pytest.raises(EncodingError):
        enc.encode(np.zeros((2, 6)))  # zero norm
    with pytest.raises(EncodingError):
        AmplitudeEncoder(n_qubits=2).encode(np.ones((2, 5)))  # too many feats


def test_qsvm_pennylane_backend_matches_numpy_backend():
    """QSVM states computed on PennyLane == states on the batched simulator."""
    from qednet.models.quantum.qsvm import QSVM
    rng = np.random.default_rng(21)
    X = rng.normal(size=(6, 6))
    q_np = QSVM(n_qubits=6, encoding="angle", backend="numpy")
    q_pl = QSVM(n_qubits=6, encoding="angle", backend="pennylane")
    s_np, s_pl = q_np._states(X), q_pl._states(X)
    assert np.allclose(np.abs(s_np), np.abs(s_pl), atol=1e-10)


def test_circuit_depth_accounting():
    from qednet.sim.statevector import circuit_depth
    gc = {"ry": 16, "cnot": 8}
    d = circuit_depth(gc, n_qubits=4)
    assert d >= 2
    assert isinstance(d, int)
