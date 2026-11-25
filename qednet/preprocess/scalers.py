"""Scaler implementations."""

from numpy.typing import NDArray
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from qednet.models.base import BaseTransformer


class StandardScaler(BaseTransformer):
    """Standard scaler wrapper."""

    def __init__(self):
        self._scaler = SklearnStandardScaler()

    def fit(self, X: NDArray, y: NDArray | None = None) -> "StandardScaler":
        self._scaler.fit(X)
        return self

    def transform(self, X: NDArray) -> NDArray:
        return self._scaler.transform(X)


class RobustScaler(BaseTransformer):
    """Robust scaler wrapper."""

    def __init__(self):
        self._scaler = SklearnRobustScaler()

    def fit(self, X: NDArray, y: NDArray | None = None) -> "RobustScaler":
        self._scaler.fit(X)
        return self

    def transform(self, X: NDArray) -> NDArray:
        return self._scaler.transform(X)
