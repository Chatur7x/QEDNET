"""Entanglement diagnostics (D7)."""

from __future__ import annotations

import numpy as np
import pennylane as qml


def _reduced_density_matrix(
    statevector: np.ndarray, subsystem: list[int], n_qubits: int
) -> np.ndarray:
    state = np.asarray(statevector).reshape([2] * n_qubits)
    all_qubits = set(range(n_qubits))
    trace_out = sorted(all_qubits - set(subsystem))
    rho = np.tensordot(state, state.conj(), axes=(trace_out, trace_out))
    tr = np.trace(rho)
    if tr > 0:
        rho = rho / tr
    return rho


def entanglement_entropy(statevector: np.ndarray, subsystem: list[int], n_qubits: int) -> float:
    rho = _reduced_density_matrix(statevector, subsystem, n_qubits)
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log2(evals)))


def meyer_wallach(statevector: np.ndarray, n_qubits: int) -> float:
    total = 0.0
    for i in range(n_qubits):
        rho_i = _reduced_density_matrix(statevector, [i], n_qubits)
        total += 1.0 - np.trace(rho_i @ rho_i)
    return float(2.0 * total / n_qubits)


def measure_entanglement(circuit_fn, *args, n_qubits: int, **kwargs) -> dict:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def _sv_circ(*a, **k):
        circuit_fn(*a, **k)
        return qml.state()

    sv = _sv_circ(*args, **kwargs)
    sv = np.asarray(sv)

    return {
        "entanglement_entropies": {
            f"qubit_{i}": entanglement_entropy(sv, [i], n_qubits) for i in range(n_qubits)
        },
        "meyer_wallach": meyer_wallach(sv, n_qubits),
        "n_qubits": n_qubits,
    }
