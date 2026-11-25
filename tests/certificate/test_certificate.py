"""Advantage Certificate quantum-side tests."""

import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score

from qednet.certificate import (
    DequantizationResult,
    EntanglementContribution,
    QuantumCost,
    dequantize_kernel,
    entanglement_ablation,
    quantum_cost,
)
from qednet.models.quantum import VQC


def test_entanglement_ablation():
    X = np.array([[0.5, 0.1], [-0.3, 0.8], [0.2, -0.5], [-0.1, -0.2]])
    y = np.array([1, 1, 0, 0])

    # Train a small VQC
    vqc = VQC(n_qubits=2, n_layers=1, max_iter=5, seed=0).fit(X, y)
    params = vqc.theta_

    # Circuit function as plain quantum function (not QNode)
    def circuit_fn(params, x):
        for i in range(2):
            qml.RX(x[i], wires=i)
        for layer in range(1):
            for i in range(2):
                qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
            qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(0))

    result = entanglement_ablation(circuit_fn, params, X, y, accuracy_score, n_qubits=2, n_layers=1)
    assert isinstance(result, EntanglementContribution)
    assert result.parameter_count == 6
    assert result.n_qubits == 2
    assert result.n_layers == 1


def test_dequantize_kernel():
    # Simple quantum kernel (2x2)
    K = np.array([[1.0, 0.3], [0.3, 1.0]])
    result = dequantize_kernel(K, n_rff=10, n_nystrom=5)
    assert isinstance(result, DequantizationResult)
    assert result.n_features == 2
    assert result.rff_error is not None
    assert result.nystrom_error is not None
    assert result.rff_kernel.shape == (2, 2)
    assert result.nystrom_kernel.shape == (2, 2)


def test_quantum_cost():
    stats = {
        "n_qubits": 4,
        "depth": 10,
        "n_params": 24,
        "gate_counts": {"total": 30, "one_qubit": 24, "two_qubit": 6},
    }
    cost = quantum_cost(stats, shots=1024, runtime_s=1.5)
    assert isinstance(cost, QuantumCost)
    assert cost.n_qubits == 4
    assert cost.depth == 10
    assert cost.two_qubit_gates == 6
    assert cost.shots == 1024
    assert cost.qpu_seconds is not None
