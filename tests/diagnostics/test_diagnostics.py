"""Quantum diagnostics tests."""

import numpy as np
import pennylane as qml

from qednet.diagnostics import (
    circuit_stats,
    gradient_variance,
    barren_plateau_diagnostic,
    entanglement_entropy,
    meyer_wallach,
    measure_entanglement,
)


def test_circuit_stats():
    def circ(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(x[2], wires=2)
        qml.CNOT(wires=[1, 2])

    stats = circuit_stats(circ, np.array([0.1, 0.2, 0.3]))
    assert stats["n_qubits"] == 3
    assert stats["gate_counts"]["total"] == 5
    assert stats["gate_counts"]["two_qubit"] == 2
    assert stats["n_params"] == 3


def test_gradient_variance():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circ(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(0))

    params = np.array([0.5, -0.3])
    res = gradient_variance(circ, params)
    assert "variance" in res
    assert res["n_params"] == 2
    assert res["variance"] >= 0


def test_barren_plateau_diagnostic():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circ(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(0))

    params = np.array([0.5, -0.3])
    res = barren_plateau_diagnostic(circ, params, n_samples=5)
    assert "mean_variance" in res
    assert res["n_samples"] == 5
    assert "criterion" in res
    assert "note" in res


def test_entanglement_measures():
    # Product state |00> -> entropy 0
    sv0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    assert abs(entanglement_entropy(sv0, [0], 2)) < 1e-10
    assert abs(entanglement_entropy(sv0, [1], 2)) < 1e-10
    assert abs(meyer_wallach(sv0, 2)) < 1e-10

    # Bell state (|00> + |11>)/sqrt(2) -> entropy 1
    sv1 = np.array([1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)], dtype=complex)
    ent = entanglement_entropy(sv1, [0], 2)
    assert abs(ent - 1.0) < 1e-6
    mw = meyer_wallach(sv1, 2)
    assert abs(mw - 1.0) < 1e-6


def test_measure_entanglement():
    def circ():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])

    res = measure_entanglement(circ, n_qubits=2)
    assert "entanglement_entropies" in res
    assert "meyer_wallach" in res
    assert res["n_qubits"] == 2
    # Bell state should have MW = 1
    assert abs(res["meyer_wallach"] - 1.0) < 1e-6