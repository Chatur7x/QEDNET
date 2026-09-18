"""Angle encoding for quantum circuits."""

import numpy as np
from numpy.typing import NDArray
from qednet.encoding.base import BaseEncoder


class AngleEncoder(BaseEncoder):
    """Angle encoding: maps features to rotation angles."""

    def __init__(self, n_qubits: int = 8, scaling: float = np.pi):
        self.n_qubits = n_qubits
        self.scaling = scaling

    def encode(self, X: NDArray) -> NDArray:
        """Encode features as rotation angles.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Encoded angles of shape (n_samples, n_qubits)
        """
        n_samples = X.shape[0]
        # Normalize to [0, 1] then scale
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
        angles = X_norm * self.scaling

        # Pad or truncate to n_qubits
        if angles.shape[1] < self.n_qubits:
            padding = np.zeros((n_samples, self.n_qubits - angles.shape[1]))
            angles = np.hstack([angles, padding])
        elif angles.shape[1] > self.n_qubits:
            angles = angles[:, :self.n_qubits]

        return angles

    def get_circuit_params(self, X: NDArray) -> NDArray:
        """Get circuit parameters (same as encode for angle encoding)."""
        return self.encode(X)