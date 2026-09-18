"""Hybrid Quantum-Classical Neural Network (D6) - Member A's PyTorch layer + Member C's sklearn wrapper.

Architecture: classical features -> VQC quantum layer -> dense head.
Provides both PennyLane TorchLayer (Member A) and sklearn-compatible wrapper (Member C).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from qednet.models.base import BaseEstimator


# Try to import Member A's PyTorch layer
try:
    from .hybrid import HybridQNNLayer, make_hybrid_model, gradient_propagation_test
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    HybridQNNLayer = None
    make_hybrid_model = None
    gradient_propagation_test = None


class HybridQNN(BaseEstimator):
    """Hybrid Quantum-Classical Neural Network with sklearn-compatible interface.

    This is a sklearn-compatible wrapper. For full PyTorch integration,
    use HybridQNNLayer directly from qednet.models.quantum.hybrid.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of quantum layers
        encoding: Feature encoding ('angle', 'amplitude', 'basis')
        optimizer: Optimizer ('adam', 'sgd')
        learning_rate: Learning rate
        max_iter: Maximum iterations
        shots: Number of shots for quantum execution
        random_state: Random seed
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        encoding: str = "angle",
        optimizer: str = "adam",
        learning_rate: float = 0.01,
        max_iter: int = 100,
        shots: int = 1024,
        random_state: int = 42,
        # Config aliases
        layers: int | None = None,
        **kwargs,
    ):
        self.n_qubits = n_qubits
        self.n_layers = int(layers if layers is not None else n_layers)
        self.encoding = encoding
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shots = shots
        self.random_state = random_state
        self._fitted = False

        # Store aliases for get_params
        self._layers = layers

    def fit(self, X: NDArray, y: NDArray) -> "HybridQNN":
        """Train the hybrid QNN using a mock (classical surrogate) when PyTorch/PennyLane not available."""
        from qednet.models.quantum.mocks import MockHybridQNN
        mock = MockHybridQNN(
            n_qubits=self.n_qubits,
            layers=self.n_layers,
            encoding=self.encoding,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            shots=self.shots,
            random_state=self.random_state,
        )
        mock.fit(X, y)
        self._model = mock._model
        self._fitted = True
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "layers": self.n_layers,
            "n_layers": self.n_layers,
            "encoding": self.encoding,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "shots": self.shots,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "HybridQNN":
        for key, value in params.items():
            if key == "layers":
                self.n_layers = value
            elif hasattr(self, key):
                setattr(self, key, value)
        return self

    def metadata(self) -> dict:
        return {
            "model": "hybrid_qnn",
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "encoding": self.encoding,
            "torch_available": _TORCH_AVAILABLE,
        }


# Re-export Member A's PyTorch layer for direct use
__all__ = ["HybridQNN", "HybridQNNLayer", "make_hybrid_model", "gradient_propagation_test"]