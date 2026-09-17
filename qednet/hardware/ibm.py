"""IBM Quantum hardware execution layer (D10).

Uses Qiskit 1.x + qiskit-ibm-runtime.
Credentials from environment/config only - never hard-coded.
Dry-run/local mode works without IBM credentials.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class IBMCredentials:
    """IBM Quantum credentials from environment."""
    token: Optional[str] = None
    instance: Optional[str] = None
    channel: str = "ibm_quantum"

    @classmethod
    def from_env(cls) -> "IBMCredentials":
        return cls(
            token=os.environ.get("IBM_QUANTUM_TOKEN"),
            instance=os.environ.get("IBM_QUANTUM_INSTANCE"),
            channel=os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum"),
        )

    def is_configured(self) -> bool:
        return self.token is not None and len(self.token) > 0


@dataclass
class BackendInfo:
    """IBM backend information."""
    name: str
    n_qubits: int
    pending_jobs: int
    status: str
    version: int
    coupling_map: list = field(default_factory=list)
    basis_gates: list = field(default_factory=list)
    max_shots: int = 8192
    simulator: bool = False


@dataclass
class HardwareResult:
    """Result from IBM hardware execution."""
    backend: str
    shots: int
    counts: dict
    runtime_s: float
    transpiled_depth: int
    transpiled_gates: dict
    error_rates: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    job_id: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class IBMExecutor:
    """IBM Quantum hardware executor with local dry-run support."""

    def __init__(
        self,
        credentials: Optional[IBMCredentials] = None,
        backend_name: Optional[str] = None,
        shots: int = 1024,
        dry_run: bool = True,
        optimization_level: int = 3,
    ):
        self.credentials = credentials or IBMCredentials.from_env()
        self.backend_name = backend_name
        self.shots = int(shots)
        self.dry_run = dry_run or not self.credentials.is_configured()
        self.optimization_level = optimization_level
        self._service = None
        self._backend = None

    def _get_service(self):
        """Lazy-load Qiskit Runtime service."""
        if self._service is None and not self.dry_run:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                self._service = QiskitRuntimeService(
                    channel=self.credentials.channel,
                    token=self.credentials.token,
                    instance=self.credentials.instance,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize IBM Runtime service: {e}")
        return self._service

    def _get_backend(self):
        """Get backend, selecting least busy if not specified."""
        if self._backend is not None:
            return self._backend

        if self.dry_run:
            # Local simulator - use basic AerSimulator
            from qiskit_aer import AerSimulator
            self._backend = AerSimulator()
            return self._backend

        service = self._get_service()
        if self.backend_name:
            self._backend = service.backend(self.backend_name)
        else:
            # Select least busy operational backend
            backends = service.backends(
                operational=True,
                simulator=False,
                min_num_qubits=5,
            )
            if not backends:
                raise RuntimeError("No suitable IBM backends available")
            self._backend = min(backends, key=lambda b: b.status().pending_jobs)
        return self._backend

    def list_backends(self) -> list[BackendInfo]:
        """List available backends with info."""
        if self.dry_run:
            return [BackendInfo(
                name="local_simulator", n_qubits=5, pending_jobs=0,
                status="active", version=1, simulator=True,
            )]

        service = self._get_service()
        infos = []
        for b in service.backends(operational=True, simulator=False):
            try:
                status = b.status()
                cfg = b.configuration()
                infos.append(BackendInfo(
                    name=b.name,
                    n_qubits=cfg.n_qubits,
                    pending_jobs=status.pending_jobs,
                    status=status.status_msg,
                    version=cfg.version,
                    coupling_map=list(cfg.coupling_map) if cfg.coupling_map else [],
                    basis_gates=list(cfg.basis_gates) if cfg.basis_gates else [],
                    max_shots=getattr(cfg, "max_shots", 8192),
                    simulator=False,
                ))
            except Exception:
                continue
        return infos

    def transpile_circuit(self, circuit) -> tuple:
        """Transpile a PennyLane/Qiskit circuit for the target backend."""
        from qiskit import transpile

        backend = self._get_backend()

        # Convert PennyLane to Qiskit if needed
        if hasattr(circuit, "to_qiskit"):
            qc = circuit.to_qiskit()
        elif hasattr(circuit, "circuit"):
            qc = circuit.circuit()
        else:
            qc = circuit

        # Transpile
        t0 = time.time()
        transpiled = transpile(
            qc,
            backend=backend,
            optimization_level=self.optimization_level,
            scheduling_method="asap" if not self.dry_run else None,
        )
        transpile_time = time.time() - t0

        # Count gates
        gate_counts = dict(transpiled.count_ops())
        return transpiled, {
            "depth": transpiled.depth(),
            "gate_counts": gate_counts,
            "transpile_time_s": transpile_time,
            "num_qubits": transpiled.num_qubits,
        }

    def execute(self, circuit, params: Optional[np.ndarray] = None) -> HardwareResult:
        """Execute circuit on IBM hardware or simulator."""
        t0 = time.time()

        # Handle parameter binding if needed
        if params is not None:
            if hasattr(circuit, "bind_parameters"):
                circuit = circuit.bind_parameters(params)
            elif hasattr(circuit, "assign_parameters"):
                circuit = circuit.assign_parameters(params)

        # Transpile
        transpiled, transpile_info = self.transpile_circuit(circuit)

        if self.dry_run:
            # Local simulation using qiskit-aer
            result = self._local_simulate(transpiled)
            runtime = time.time() - t0
            return HardwareResult(
                backend="local_simulator",
                shots=self.shots,
                counts=result,
                runtime_s=runtime,
                transpiled_depth=transpile_info["depth"],
                transpiled_gates=transpile_info["gate_counts"],
                success=True,
            )

        # Real hardware execution
        try:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from qiskit_ibm_runtime import Session

            backend = self._get_backend()
            with Session(backend=backend) as session:
                sampler = Sampler(mode=session)
                job = sampler.run([transpiled], shots=self.shots)
                job_result = job.result()
                job_id = job.job_id()

            # Extract counts
            counts = {}
            if hasattr(job_result, "quasi_dists"):
                for i, qd in enumerate(job_result.quasi_dists):
                    counts = {k: int(v * self.shots) for k, v in qd.items()}
            elif hasattr(job_result, "get_counts"):
                counts = job_result.get_counts()

            # Get error rates if available
            error_rates = {}
            calib = {}
            try:
                props = backend.properties()
                if props:
                    for i in range(backend.configuration().n_qubits):
                        error_rates[f"qubit_{i}_readout"] = props.readout_error(i)
                    for gate in props.gates:
                        error_rates[f"{gate.gate}_error"] = gate.parameters[0].value if gate.parameters else 0
            except Exception:
                pass

            runtime = time.time() - t0
            return HardwareResult(
                backend=backend.name,
                shots=self.shots,
                counts=counts,
                runtime_s=runtime,
                transpiled_depth=transpile_info["depth"],
                transpiled_gates=transpile_info["gate_counts"],
                error_rates=error_rates,
                calibration=calib,
                job_id=job_id,
                success=True,
            )

        except Exception as e:
            return HardwareResult(
                backend=self.backend_name or "unknown",
                shots=self.shots,
                counts={},
                runtime_s=time.time() - t0,
                transpiled_depth=transpile_info.get("depth", 0),
                transpiled_gates=transpile_info.get("gate_counts", {}),
                success=False,
                error=str(e),
            )

    def _local_simulate(self, circuit) -> dict:
        """Local simulation fallback."""
        try:
            from qiskit_aer import AerSimulator
            sim = AerSimulator()
        except Exception:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            # Very basic simulation
            sv = Statevector.from_instruction(circuit)
            probs = sv.probabilities()
            counts = {}
            for i, p in enumerate(probs):
                if p > 1e-10:
                    counts[format(i, f"0{circuit.num_qubits}b")] = int(p * self.shots)
            return counts

        from qiskit import transpile
        tcirc = transpile(circuit, sim)
        job = sim.run(tcirc, shots=self.shots)
        return job.result().get_counts()


def create_executor(
    backend: Optional[str] = None,
    shots: int = 1024,
    dry_run: bool = True,
    optimization_level: int = 3,
) -> IBMExecutor:
    """Factory for IBMExecutor with environment credentials."""
    return IBMExecutor(
        backend_name=backend,
        shots=shots,
        dry_run=dry_run,
        optimization_level=optimization_level,
    )