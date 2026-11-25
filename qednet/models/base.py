"""Base interfaces for estimators and transformers."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseEstimator(ABC):
    """Base interface for all models (quantum and classical).

    Every model must implement:
        fit(X, y)
        predict_proba(X)
    """

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> "BaseEstimator":
        """Fit the model to training data.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)

        Returns:
            self
        """
        ...

    @abstractmethod
    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Probabilities of shape (n_samples, n_classes)
        """
        ...

    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator (introspected; sklearn-cloneable)."""
        import inspect

        try:
            sig = inspect.signature(self.__init__)
        except (TypeError, ValueError):
            return {}
        return {
            p: getattr(self, p) for p in sig.parameters if p not in ("kwargs",) and hasattr(self, p)
        }

    def set_params(self, **params: Any) -> "BaseEstimator":
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class BaseTransformer(ABC):
    """Base interface for all preprocessing transformers.

    Every transformer must implement:
        fit(X, y=None)
        transform(X)
    """

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray | None = None) -> "BaseTransformer":
        """Fit the transformer to data.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Optional training labels

        Returns:
            self
        """
        ...

    @abstractmethod
    def transform(self, X: NDArray) -> NDArray:
        """Transform the data.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Transformed features
        """
        ...

    def fit_transform(self, X: NDArray, y: NDArray | None = None) -> NDArray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> "BaseTransformer":
        for key, value in params.items():
            setattr(self, key, value)
        return self
