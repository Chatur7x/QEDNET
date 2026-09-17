"""A. Parameter-matched entanglement ablation (D8).

Replaces each two-qubit entangling gate with trainable single-qubit rotations
to preserve parameter count. Returns structured comparison results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


@dataclass
class EntanglementContribution:
    """Result of parameter-matched entanglement ablation."""
    with_entanglement: float
    without_entanglement: float
    parameter_count: int
    n_qubits: int
    n_layers: int
    delta: float
    relative_drop: float

    def __str__(self):
        return (
            f"EntanglementContribution(delta={self.delta:.4f}, "
            f"rel_drop={self.relative_drop:.2%}, params={self.parameter_count})"
        )


def _make_vqc_circuit(n_qubits: int, n_layers: int, use_entanglement: bool = True) -> Callable:
    """Factory for VQC-like circuit with/without entanglement but same param count."""

    def circuit(params, x):
        for i in range(min(len(x), n_qubits)):
            qml.RX(x[i], wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
            if use_entanglement:
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])
        return qml.expval(qml.Z(0))

    return circuit


def entanglement_ablation(
    circuit_fn: Callable,
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable,
    n_qubits: int,
    n_layers: int,
    device: str = "default.qubit",
    shots: int | None = None,
) -> EntanglementContribution:
    """Run parameter-matched entanglement ablation.

    Args:
        circuit_fn: function(params, x) -> expectation
        params: trained parameters (n_layers, n_qubits, 3)
        X: feature matrix
        y: labels
        metric_fn: function(y_true, y_pred) -> float
        n_qubits: number of qubits
        n_layers: number of layers
        device: PennyLane device
        shots: shots for hardware-compatible evaluation

    Returns:
        EntanglementContribution with structured results
    """
    kwargs = {"wires": n_qubits}
    if shots is not None:
        kwargs["shots"] = shots
    try:
        dev = qml.device(device, **kwargs)
    except Exception:
        dev = qml.device("default.qubit", **kwargs)

    @qml.qnode(dev)
    def circ_with(params, x):
        return circuit_fn(params, x)

    @qml.qnode(dev)
    def circ_without(params, x):
        for i in range(min(len(x), n_qubits)):
            qml.RX(x[i], wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
            for i in range(n_qubits - 1):
                qml.RZ(params[layer, i, 0], wires=i)
                qml.RY(params[layer, i, 1], wires=i + 1)
            if n_qubits > 2:
                qml.RZ(params[layer, -1, 0], wires=n_qubits - 1)
                qml.RY(params[layer, 0, 1], wires=0)
        return qml.expval(qml.Z(0))

    preds_with = np.array([circ_with(params, x) for x in X])
    preds_without = np.array([circ_without(params, x) for x in X])

    with_metric = metric_fn(y, (preds_with > 0).astype(int))
    without_metric = metric_fn(y, (preds_without > 0).astype(int))

    param_count = n_layers * n_qubits * 3
    delta = with_metric - without_metric
    rel = delta / with_metric if with_metric != 0 else 0.0

    return EntanglementContribution(
        with_entanglement=with_metric,
        without_entanglement=without_metric,
        parameter_count=param_count,
        n_qubits=n_qubits,
        n_layers=n_layers,
        delta=delta,
        relative_drop=rel,
    )


def vqc_ablation_cost(vqc_model, X_test):
    """Extract cost from a trained VQC model."""
    from .costing import quantum_cost
    meta = vqc_model.metadata()
    return quantum_cost(
        circuit_stats={
            "n_qubits": meta["n_qubits"],
            "depth": meta.get("circuit_depth", meta["n_layers"] * 3),
            "n_params": meta["n_params"],
            "gate_counts": {
                "total": meta["n_params"] + meta.get("two_qubit_gates", 0),
                "one_qubit": meta["n_params"],
                "two_qubit": meta.get("two_qubit_gates", 0),
            },
        },
        shots=meta.get("shots", 1024) or 1024,
        runtime_s=meta.get("fit_time_s", 0.0),
    )


def qsvm_ablation_cost(qsvm_model):
    """Extract cost from a trained QSVM model."""
    from .costing import quantum_cost
    meta = qsvm_model.metadata()
    n_evals = meta.get("kernel_evals", 0)
    return quantum_cost(
        circuit_stats={
            "n_qubits": meta["n_qubits"],
            "depth": meta.get("circuit_depth", 3),
            "n_params": 0,
            "gate_counts": {
                "total": n_evals * meta.get("two_qubit_gates", 0),
                "one_qubit": n_evals * (meta["n_qubits"] + 1),
                "two_qubit": n_evals * meta.get("two_qubit_gates", 0),
            },
        },
        shots=meta.get("shots", 1024) or 1024,
        runtime_s=meta.get("fit_time_s", 0.0),
    )