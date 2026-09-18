"""Data re-uploading encoding: features re-injected at every layer.

Unlike one-shot encoders, re-uploading interleaves the features with
trainable quantum layers. The encoder therefore exposes per-layer angle
schedules consumed by the DataReuploading model; ``encode`` returns the
raw base angles (scaled features) for interface consistency.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseEncoder, EncodedBatch


class ReuploadEncoder(BaseEncoder):
    """Feature re-injection schedule for data-re-uploading circuits.

    ``n_qubits`` is configurable (default 8); each qubit q receives feature
    ``x[q % n_features]`` scaled into [0, pi] at every layer, so layers can
    each modulate the re-injected signal with trainable weights.
    """
    name = "reupload"

    def n_qubits_for(self, n_features: int) -> int:
        return int(min(max(2, n_features), 8))

    def encode(self, X: np.ndarray) -> EncodedBatch:
        X = np.asarray(X, dtype=np.float64)
        nq = self._n_qubits or self.n_qubits_for(X.shape[1])
        self.validate(X, nq)
        Xc = np.clip(X, 0.0, 1.0)
        # per-qubit base angles: feature q %% n_features
        idx = np.arange(nq) % X.shape[1]
        angles = np.pi * Xc[:, idx]
        info: Dict[str, Any] = {
            "encoder": self.name, "n_qubits": nq,
            "feature_assignment": "qubit q <- feature q mod n_features",
            "re_injection": "per trainable layer",
            "shots": None,
        }
        return EncodedBatch(angles=angles, n_qubits=nq, info=info)

    def layer_angles(self, X: np.ndarray, layer: int) -> np.ndarray:
        """Base angles for layer ``layer`` (identical schedule per layer)."""
        return self.encode(X).angles
