"""Basis encoding for quantum circuits."""

import numpy as np
from numpy.typing import NDArray

from qednet.encoding.base import BaseEncoder


class BasisEncoder(BaseEncoder):
    """Basis encoding: maps features to computational basis states."""

    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits

    def encode(self, X: NDArray) -> NDArray:
        """Encode features as basis state indices.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Basis state indices of shape (n_samples,)
        """
        # For basis encoding, we discretize features to basis states
        # This is a simplified version - in practice would use more sophisticated mapping
        n_samples = X.shape[0]
        # Use first feature to determine basis state (simplified)
        feature = X[:, 0] if X.shape[1] > 0 else np.zeros(n_samples)
        # Discretize to 2^n_qubits states
        n_states = 2**self.n_qubits
        indices = np.clip(
            (
                (feature - feature.min()) / (feature.max() - feature.min() + 1e-8) * (n_states - 1)
            ).astype(int),
            0,
            n_states - 1,
        )
        return indices

    def get_circuit_params(self, X: NDArray) -> NDArray:
        """Get circuit parameters (basis state indices)."""
        return self.encode(X)
