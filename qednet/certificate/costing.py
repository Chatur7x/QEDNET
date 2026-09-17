"""C. Quantum cost accounting (D8).

Measures/exposes:
- qubits
- depth
- gates
- two-qubit gates
- parameters
- shots
- runtime
- estimated quantum execution cost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QuantumCost:
    """Quantum execution cost breakdown."""
    n_qubits: int
    depth: int
    total_gates: int
    one_qubit_gates: int
    two_qubit_gates: int
    n_params: int
    shots: int
    runtime_s: float
    # Estimated cost on hardware (simplified)
    qpu_seconds: Optional[float] = None  # estimated QPU time
    cost_usd: Optional[float] = None     # estimated USD cost

    def __str__(self):
        return (
            f"QuantumCost(qubits={self.n_qubits}, depth={self.depth}, "
            f"2q_gates={self.two_qubit_gates}, params={self.n_params}, "
            f"shots={self.shots}, runtime={self.runtime_s:.3f}s)"
        )


def quantum_cost(
    circuit_stats: dict,
    shots: int = 1024,
    runtime_s: float = 0.0,
    qpu_time_per_shot: float = 1e-6,  # ~1 microsec per shot on superconducting
    cost_per_qpu_second: float = 0.0,  # $/sec, 0 = unknown
) -> QuantumCost:
    """Compute quantum cost from circuit stats and execution parameters."""
    gc = circuit_stats.get("gate_counts", {})
    n_qubits = circuit_stats.get("n_qubits", 0)
    depth = circuit_stats.get("depth", 0)
    n_params = circuit_stats.get("n_params", 0)

    total_gates = gc.get("total", 0)
    one_q = gc.get("one_qubit", 0)
    two_q = gc.get("two_qubit", 0)

    # Estimate QPU time: circuit_time + measurement_time
    # Circuit time ~ depth * gate_time (gate_time ~ 50ns for 1q, 200ns for 2q)
    gate_time_1q = 50e-9
    gate_time_2q = 200e-9
    circuit_time = (gc.get("one_qubit", 0) * gate_time_1q +
                    gc.get("two_qubit", 0) * gate_time_2q)
    meas_time = shots * qpu_time_per_shot
    est_qpu = circuit_time + meas_time

    cost = None
    if cost_per_qpu_second > 0:
        cost = est_qpu * cost_per_qpu_second

    return QuantumCost(
        n_qubits=n_qubits,
        depth=depth,
        total_gates=total_gates,
        one_qubit_gates=one_q,
        two_qubit_gates=two_q,
        n_params=n_params,
        shots=shots,
        runtime_s=runtime_s,
        qpu_seconds=est_qpu,
        cost_usd=cost,
    )