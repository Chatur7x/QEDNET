"""Base encoder interface."""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class BaseEncoder(ABC):
    """Base interface for quantum encoders."""

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