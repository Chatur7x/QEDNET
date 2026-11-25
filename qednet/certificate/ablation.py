"""A. Parameter-matched entanglement ablation (D8)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pennylane as qml


@dataclass
class EntanglementContribution:
    with_entanglement: float
    without_entanglement: float
    parameter_count: int
    n_qubits: int
    n_layers: int
    delta: float
    relative_drop: float


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
