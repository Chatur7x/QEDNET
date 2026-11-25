"""Amplitude encoding for quantum circuits."""

import numpy as np
from numpy.typing import NDArray

from qednet.encoding.base import BaseEncoder


class AmplitudeEncoder(BaseEncoder):
    """Amplitude encoding: maps features to quantum state amplitudes."""

    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits

    def encode(self, X: NDArray) -> NDArray:
        """Encode features as statevector amplitudes.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Normalized amplitudes of shape (n_samples, 2^n_qubits)
        """
        n_samples = X.shape[0]
        dim = 2**self.n_qubits

        # Pad or truncate features
        if X.shape[1] < dim:
            padding = np.zeros((n_samples, dim - X.shape[1]))
            X_padded = np.hstack([X, padding])
        else:
            X_padded = X[:, :dim]

        # Normalize to unit vector
        norms = np.linalg.norm(X_padded, axis=1, keepdims=True)
        amplitudes = X_padded / (norms + 1e-12)

        return amplitudes

    def get_circuit_params(self, X: NDArray) -> NDArray:
        """Get circuit parameters for amplitude encoding."""
        return self.encode(X)
