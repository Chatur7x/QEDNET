"""Angle encoding: one RY(pi * x_q) rotation per feature/qubit."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseEncoder, EncodedBatch, EncodingError


class AngleEncoder(BaseEncoder):
    """RY angle encoding. ``n_qubits == n_features``.

    Features are expected in [0, 1] (the pipeline's sigmoid-scaled output);
    angles are pi * x so a full rotation range is available. Values outside
    [0, 1] are accepted but clipped to [0, 1] with a warning flag, because
    encoding periodicity silently aliases otherwise.
    """
    name = "angle"

    def n_qubits_for(self, n_features: int) -> int:
        return int(n_features)

    def encode(self, X: np.ndarray) -> EncodedBatch:
        X = np.asarray(X, dtype=np.float64)
        nq = self._n_qubits or self.n_qubits_for(X.shape[1])
        self.validate(X, nq)
        if X.shape[1] != nq:
            raise EncodingError(
                f"angle: feature dim {X.shape[1]} != qubit count {nq} "
                "(angle encoding requires one feature per qubit)")

        clipped = int(np.sum((X < 0) | (X > 1)))
        Xc = np.clip(X, 0.0, 1.0)
        angles = np.pi * Xc
        info: Dict[str, Any] = {
            "encoder": self.name,
            "n_qubits": nq,
            "angle_scale": "pi * x",
            "clipped_values": clipped,
            "shots": None,
        }
        return EncodedBatch(angles=angles, n_qubits=nq, info=info)

    def states(self, X: np.ndarray) -> np.ndarray:
        """Exact feature-map states |phi(x)> for kernel computation."""
        batch = self.encode(X)
        nq = batch.n_qubits
        B = X.shape[0]
        state = self._zero_states(nq, B)
        from ..sim.statevector import ry_matrix, apply_gate_2x2
        for q in range(nq):
            G = ry_matrix(batch.angles[:, q]).astype(np.complex128)
            state = apply_gate_2x2(state, G, q)
        flat = state.reshape(B, -1)
        return flat
