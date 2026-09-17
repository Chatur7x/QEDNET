from __future__ import annotations

"""Base encoder interface and shared validation utilities (D4 + Member C)."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseEncoder(ABC):
    """Base interface for quantum encoders (Member C class-based API)."""

    @abstractmethod
    def encode(self, X: NDArray) -> NDArray:
        """Encode classical data into quantum-compatible format."""
        ...

    @abstractmethod
    def get_circuit_params(self, X: NDArray) -> NDArray:
        """Get circuit parameters for encoded data."""
        ...

    def __call__(self, X: NDArray) -> NDArray:
        return self.encode(X)


# Member A's validation utilities (D4)
def validate_vector(x, n_qubits: int, encoding: str) -> np.ndarray:
    """Validate a single feature vector for the given encoding."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{encoding}: empty feature vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{encoding}: features must be finite")
    if n_qubits < 1:
        raise ValueError(f"{encoding}: n_qubits must be >= 1, got {n_qubits}")
    if encoding in ("angle", "zz", "reupload"):
        if arr.size > n_qubits:
            raise ValueError(
                f"{encoding}: n_features={arr.size} > n_qubits={n_qubits}; "
                "compress upstream instead of truncating"
            )
    elif encoding == "amplitude":
        if arr.size > 2**n_qubits:
            raise ValueError(
                f"amplitude: n_features={arr.size} > 2**n_qubits={2 ** n_qubits}"
            )
    else:
        raise ValueError(f"unknown encoding: {encoding}")
    return arr


def pad_to_pow2(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """Zero-pad a vector to 2**n_qubits and L2-normalize for amplitude encoding."""
    dim = 2**n_qubits
    v = np.zeros(dim)
    v[:x.size] = x
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("amplitude: zero vector cannot be normalized")
    return v / norm