"""Circuit statistics (D7).

Exposes:
- qubit count
- circuit depth
- total gate count
- one-qubit gate count
- two-qubit gate count
- trainable parameter count
"""

from __future__ import annotations

import pennylane as qml


def count_gates(tape: qml.tape.QuantumTape) -> dict:
    """Count gates by type from a PennyLane tape."""
    counts = {"total": 0, "one_qubit": 0, "two_qubit": 0, "multi_qubit": 0}
    for op in tape.operations:
        counts["total"] += 1
        n_wires = len(op.wires)
        if n_wires == 1:
            counts["one_qubit"] += 1
        elif n_wires == 2:
            counts["two_qubit"] += 1
        else:
            counts["multi_qubit"] += 1
    return counts


def circuit_stats(quantum_fn, *args, **kwargs) -> dict:
    """Extract circuit statistics from a plain quantum function.

    Args:
        quantum_fn: plain function applying quantum gates (not a QNode)
        *args, **kwargs: arguments passed to quantum_fn

    Returns dict with:
    - n_qubits
    - depth
    - n_params
    - gate_counts: {total, one_qubit, two_qubit, multi_qubit}
    - operations: list of (name, wires) for inspection
    """
    # Use make_tape to extract the circuit structure from quantum function
    tape = qml.transforms.make_tape(quantum_fn)(*args, **kwargs)

    # Depth: longest path in DAG
    try:
        depth = qml.transforms.depth(tape)
    except Exception:
        depth = len(tape.operations)

    gate_counts = count_gates(tape)

    # Count parameters from operations
    n_params = 0
    for op in tape.operations:
        if hasattr(op, "parameters") and op.parameters:
            for p in op.parameters:
                if hasattr(p, "numel"):
                    n_params += p.numel()
                elif hasattr(p, "__len__"):
                    n_params += len(p)
                else:
                    n_params += 1

    return {
        "n_qubits": len(tape.wires),
        "depth": depth,
        "n_params": n_params,
        "gate_counts": gate_counts,
        "operations": [(op.name, [int(w) for w in op.wires]) for op in tape.operations],
    }