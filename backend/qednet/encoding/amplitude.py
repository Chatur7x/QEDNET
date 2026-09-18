"""Amplitude encoding: features become statevector amplitudes."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseEncoder, EncodedBatch, EncodingError


class AmplitudeEncoder(BaseEncoder):
    """Amplitude encoding: |phi(x)> = normalised padded feature vector.

    ``n_qubits = ceil(log2(n_features))``; features are zero-padded to
    ``2**n_qubits`` and L2-normalised. Requires a non-zero feature norm.
    """
    name = "amplitude"

    def n_qubits_for(self, n_features: int) -> int:
        if n_features < 1:
            raise EncodingError("amplitude: needs >= 1 feature")
        return int(max(1, np.ceil(np.log2(max(n_features, 2)))))

    def encode(self, X: np.ndarray) -> EncodedBatch:
        X = np.asarray(X, dtype=np.float64)
        nq = self._n_qubits or self.n_qubits_for(X.shape[1])
        self.validate(X, nq)
        dim = 2 ** nq
        if X.shape[1] > dim:
            raise EncodingError(
                f"amplitude: {X.shape[1]} features exceed 2**{nq}={dim} "
                "amplitude capacity (increase n_qubits or compress more)")
        norms = np.linalg.norm(X, axis=1)
        if np.any(norms < 1e-12):
            raise EncodingError(
                "amplitude: zero-norm feature vector cannot be encoded")

        B = X.shape[0]
        padded = np.zeros((B, dim))
        padded[:, : X.shape[1]] = X
        states = padded / norms[:, None]

        # placeholder angle payload (amplitude encoding is state-based)
        angles = np.zeros((B, nq))
        info: Dict[str, Any] = {
            "encoder": self.name, "n_qubits": nq,
            "amplitude_dim": dim, "padded": dim - X.shape[1],
            "shots": None,
        }
        return EncodedBatch(angles=angles, n_qubits=nq, states=states,
                            info=info)

    def states(self, X: np.ndarray) -> np.ndarray:
        return self.encode(X).states
