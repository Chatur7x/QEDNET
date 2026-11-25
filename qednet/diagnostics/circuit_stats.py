"""Circuit statistics (D7)."""

from __future__ import annotations

import pennylane as qml


def count_gates(tape: qml.tape.QuantumTape) -> dict:
    counts = {"total": 0, "one_qubit": 0, "two_qubit": 0}
    for op in tape.operations:
        counts["total"] += 1
        n_wires = len(op.wires)
        if n_wires == 1:
            counts["one_qubit"] += 1
        elif n_wires == 2:
            counts["two_qubit"] += 1
    return counts


def circuit_stats(quantum_fn, *args, **kwargs) -> dict:
    tape = qml.transforms.make_tape(quantum_fn)(*args, **kwargs)

    try:
        depth = qml.transforms.depth(tape)
    except Exception:
        depth = len(tape.operations)

    gate_counts = count_gates(tape)

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
    }
