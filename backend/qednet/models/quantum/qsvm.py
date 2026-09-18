"""F5 — Quantum Support Vector Machine (QSVM).

Features -> quantum feature map -> quantum kernel -> SVM -> prediction.

The kernel is ACTUALLY generated from the quantum feature map: for each
sample the feature-map circuit's statevector |phi(x)> is computed on the
quantum simulator (batched statevector backend or PennyLane reference
backend), and the kernel entry is the fidelity
k(x, x') = |<phi(x) | phi(x')>|^2 computed from those quantum states —
mathematically identical to estimating the overlap with infinitely many
shots (exact simulation), as documented in the resource report.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.svm import SVC

from ...encoding import get_encoder
from ...models.base import BaseModel
from ...models.training import quantum_input_transform
from ...sim.statevector import resource_report


class QSVM(BaseModel):
    name = "qsvm"
    family = "quantum"

    def __init__(self, random_state: int = 42, n_qubits: int = 8,
                 layers: int = 0, encoding: str = "angle",
                 backend: str = "numpy", entanglement: bool = True,
                 C: float = 1.0, shots: Optional[int] = None, **kw):
        super().__init__(random_state)
        if encoding not in ("angle", "zz"):
            raise ValueError("QSVM supports encoding 'angle' or 'zz'")
        self.n_qubits = int(n_qubits)
        self.encoding = encoding
        self.backend = backend
        self.C = float(C)
        self.shots = shots
        self.encoder = get_encoder(encoding, n_qubits=n_qubits, reps=2)
        self.svc = SVC(kernel="precomputed", C=self.C, probability=True,
                       random_state=random_state)
        self._train_states: Optional[np.ndarray] = None
        self._state_compute_s = 0.0
        self._n_states = 0

    # ------------------------------------------------------------- states
    def _states(self, X: np.ndarray) -> np.ndarray:
        """Feature-map statevectors |phi(x)> per sample (quantum simulation)."""
        Xsq = quantum_input_transform(X)
        t0 = time.perf_counter()
        if self.backend == "pennylane":
            states = self._states_pennylane(Xsq)
        else:
            states = self.encoder.states(Xsq)
        self._state_compute_s += time.perf_counter() - t0
        self._n_states += X.shape[0]
        # re-normalise for numerical safety
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        return states / np.maximum(norms, 1e-12)

    def _states_pennylane(self, Xsq: np.ndarray) -> np.ndarray:
        """Reference implementation executed on PennyLane default.qubit."""
        import pennylane as qml
        nq = self.encoder.n_qubits_for(Xsq.shape[1])
        dev = qml.device("default.qubit", wires=nq)

        if self.encoding == "angle":
            @qml.qnode(dev)
            def circuit(x):
                for q in range(nq):
                    qml.RY(np.pi * x[q], wires=q)
                return qml.state()
        else:  # zz feature map on PennyLane
            @qml.qnode(dev)
            def circuit(x):
                for q in range(nq):
                    qml.Hadamard(wires=q)
                for _ in range(2):
                    for q in range(nq):
                        qml.RZ(2.0 * x[q], wires=q)
                    for c in range(nq):
                        t = (c + 1) % nq
                        qml.CNOT(wires=[c, t])
                        qml.RZ(2.0 * x[c] * x[t], wires=t)
                        qml.CNOT(wires=[c, t])
                return qml.state()

        return np.array([circuit(x) for x in Xsq])

    # -------------------------------------------------------------- kernel
    def quantum_kernel(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        """Kernel matrix k_ij = |<phi(x_i)|phi(x_j)>|^2 from quantum states."""
        sa, sb = self._states(Xa), self._states(Xb)
        gram = np.abs(sa.conj() @ sb.T) ** 2
        return np.real(gram)

    # ------------------------------------------------------------------ fit
    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self._train_states = self._states(X)
        K = np.abs(self._train_states.conj() @ self._train_states.T) ** 2
        K = np.real(K)
        self.svc.fit(K, y)

    def _positive_proba(self, X: np.ndarray) -> np.ndarray:
        states = self._states(X)
        K = np.real(np.abs(states.conj() @ self._train_states.T) ** 2)
        return self.svc.predict_proba(K)[:, 1]

    # ------------------------------------------------------------- metadata
    def n_parameters(self) -> int:
        # kernel method: no trainable circuit parameters; capacity = support
        return int(getattr(self.svc, "n_support_", np.array([0])).sum())

    def quantum_resources(self) -> Optional[Dict[str, Any]]:
        nq = self.encoder.n_qubits or self.n_qubits
        if self.encoding == "angle":
            gate_counts = {"ry": nq}
        else:
            # reps=2: H on all + per rep [RZ(2x) per qubit + (CX, RZ, CX) ring]
            gate_counts = {"h": nq, "rz": nq * 2 + nq * 2,
                           "cnot": nq * 2 * 2}
        depth = sum(gate_counts.values())
        return {
            "n_qubits": self.encoder.n_qubits or self.n_qubits,
            "gate_counts": gate_counts,
            "total_gates_per_circuit": sum(gate_counts.values()),
            "circuit_depth": depth,
            "n_circuits": self._n_states,
            "shots": None,
            "shots_note": ("exact statevector fidelity (equivalent to "
                           "shots=inf); kernel entries are inner products "
                           "of simulated quantum states"),
            "kernel_method": "fidelity |<phi(x)|phi(x')>|^2",
            "state_compute_time_s": round(self._state_compute_s, 4),
            "statevector_dimension": 2 ** (self.encoder.n_qubits or self.n_qubits),
        }

    def metadata(self) -> Dict[str, Any]:
        m = super().metadata()
        m.update({
            "n_qubits": self.encoder.n_qubits or self.n_qubits,
            "encoding": f"{self.encoding} feature map (reps=2)",
            "backend": self.backend,
            "kernel": "quantum fidelity kernel (precomputed)",
            "svm_C": self.C,
        })
        return m
