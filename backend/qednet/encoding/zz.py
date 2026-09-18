"""ZZ feature map (Qiskit-style): H, then per repetition
RZ(2 x_q) on every qubit and RZZ(2 x_i x_j) entanglers on a ring."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseEncoder, EncodedBatch


class ZZFeatureMap(BaseEncoder):
    """Second-order (ZZ) feature map with ``reps`` repetitions.

    Structure per repetition:
      * RZ(2 * x_q) on each qubit q
      * RZZ(2 * x_i * x_j) on ring pairs (q, q+1 mod n)
    Preceded by Hadamards on all qubits. Qubit count equals feature count.
    """
    name = "zz"

    def __init__(self, n_qubits: int | None = None, reps: int = 2):
        super().__init__(n_qubits)
        if reps < 1:
            raise ValueError("ZZFeatureMap reps must be >= 1")
        self.reps = reps

    def n_qubits_for(self, n_features: int) -> int:
        return int(n_features)

    def encode(self, X: np.ndarray) -> EncodedBatch:
        X = np.asarray(X, dtype=np.float64)
        nq = self._n_qubits or self.n_qubits_for(X.shape[1])
        self.validate(X, nq)
        if X.shape[1] != nq:
            raise ValueError(
                f"zz: feature dim {X.shape[1]} != qubit count {nq} "
                "(ZZ feature map requires one feature per qubit)")
        # "angles" are the RZ rotation angles 2*x (for circuit builders)
        angles = 2.0 * X
        info: Dict[str, Any] = {
            "encoder": self.name, "n_qubits": nq, "reps": self.reps,
            "structure": "H + [RZ(2x) + RZZ(2 xi xj) ring] * reps",
            "shots": None,
        }
        return EncodedBatch(angles=angles, n_qubits=nq, info=info)

    def states(self, X: np.ndarray) -> np.ndarray:
        """Exact feature-map states for kernel computation (complex)."""
        batch = self.encode(X)
        nq = batch.n_qubits
        B = X.shape[0]
        Xc = np.asarray(X, dtype=np.float64)

        # Hadamard on all qubits
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        from ..sim.statevector import apply_gate_2x2
        state = self._zero_states(nq, B)
        for q in range(nq):
            state = apply_gate_2x2(state, H, q)

        for _ in range(self.reps):
            # RZ(2 x_q) per qubit (per-sample angles)
            for q in range(nq):
                G = self._rz_complex(2.0 * Xc[:, q])
                state = apply_gate_2x2(state, G, q)
            # RZZ(2 x_i x_j) on ring pairs: CX, RZ, CX
            for c in range(nq):
                t = (c + 1) % nq
                gamma = 2.0 * Xc[:, c] * Xc[:, t]
                state = self._apply_cnot(state, c, t)
                G = self._rz_complex(gamma)
                state = apply_gate_2x2(state, G, t)
                state = self._apply_cnot(state, c, t)

        return state.reshape(B, -1)

    def _rz_complex(self, angles: np.ndarray) -> np.ndarray:
        half = 0.5 * np.asarray(angles, dtype=np.complex128)
        B = half.shape[0]
        G = np.zeros((B, 2, 2), dtype=np.complex128)
        G[:, 0, 0] = np.exp(-1j * half)
        G[:, 1, 1] = np.exp(1j * half)
        return G
