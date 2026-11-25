"""C. Quantum cost accounting (D8)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantumCost:
    n_qubits: int
    depth: int
    total_gates: int
    one_qubit_gates: int
    two_qubit_gates: int
    n_params: int
    shots: int
    runtime_s: float
    qpu_seconds: float | None = None
    cost_usd: float | None = None


def quantum_cost(
    circuit_stats: dict,
    shots: int = 1024,
    runtime_s: float = 0.0,
    qpu_time_per_shot: float = 1e-6,
    cost_per_qpu_second: float = 0.0,
) -> QuantumCost:
    gc = circuit_stats.get("gate_counts", {})
    n_qubits = circuit_stats.get("n_qubits", 0)
    depth = circuit_stats.get("depth", 0)
    n_params = circuit_stats.get("n_params", 0)

    one_q = gc.get("one_qubit", 0)
    two_q = gc.get("two_qubit", 0)

    circuit_time = one_q * 50e-9 + two_q * 200e-9
    meas_time = shots * qpu_time_per_shot
    est_qpu = circuit_time + meas_time

    cost = est_qpu * cost_per_qpu_second if cost_per_qpu_second > 0 else None

    return QuantumCost(
        n_qubits=n_qubits,
        depth=depth,
        total_gates=one_q + two_q,
        one_qubit_gates=one_q,
        two_qubit_gates=two_q,
        n_params=n_params,
        shots=shots,
        runtime_s=runtime_s,
        qpu_seconds=est_qpu,
        cost_usd=cost,
    )
