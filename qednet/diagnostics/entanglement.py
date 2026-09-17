"""Entanglement diagnostics (D7).

- entanglement_entropy: von Neumann entropy of subsystem
- meyer_wallach: global entanglement measure Q = 2(1 - Tr(ρ_A^2))/n

For pure states from statevector simulators.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


def _reduced_density_matrix(statevector: np.ndarray, subsystem: list[int],
                            n_qubits: int) -> np.ndarray:
    """Compute reduced density matrix by tracing out complement."""
    state = np.asarray(statevector).reshape([2] * n_qubits)
    # Trace out all qubits not in subsystem
    all_qubits = set(range(n_qubits))
    trace_out = sorted(all_qubits - set(subsystem))
    # Use numpy's tensordot for partial trace
    # Reshape and trace iteratively
    rho = np.tensordot(state, state.conj(), axes=(trace_out, trace_out))
    # Normalize
    tr = np.trace(rho)
    if tr > 0:
        rho = rho / tr
    return rho


def entanglement_entropy(statevector: np.ndarray, subsystem: list[int],
                         n_qubits: int) -> float:
    """Von Neumann entropy S(ρ_A) = -Tr(ρ_A log ρ_A) for subsystem."""
    rho = _reduced_density_matrix(statevector, subsystem, n_qubits)
    # Eigenvalues of reduced density matrix
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]  # ignore numerical zeros
    return float(-np.sum(evals * np.log2(evals)))


def meyer_wallach(statevector: np.ndarray, n_qubits: int) -> float:
    """Global entanglement measure Q = 2/n * sum_A (1 - Tr(ρ_A^2)).

    Q = 0 for product states, Q = 1 for maximally entangled (GHZ-like).
    """
    total = 0.0
    for i in range(n_qubits):
        rho_i = _reduced_density_matrix(statevector, [i], n_qubits)
        total += 1.0 - np.trace(rho_i @ rho_i)
    return float(2.0 * total / n_qubits)


def measure_entanglement(circuit_fn, *args, n_qubits: int, **kwargs) -> dict:
    """Run circuit, get statevector, compute entanglement measures.

    circuit_fn: a plain quantum function (not a QNode) that applies gates.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def _sv_circ(*a, **k):
        circuit_fn(*a, **k)
        return qml.state()

    sv = _sv_circ(*args, **kwargs)
    sv = np.asarray(sv)

    # Entropy for each qubit bipartition
    entropies = {}
    for i in range(n_qubits):
        entropies[f"qubit_{i}"] = entanglement_entropy(sv, [i], n_qubits)

    # Meyer-Wallach global measure
    mw = meyer_wallach(sv, n_qubits)

    return {
        "entanglement_entropies": entropies,
        "meyer_wallach": mw,
        "n_qubits": n_qubits,
    }