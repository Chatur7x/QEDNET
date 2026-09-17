"""Scaler implementations."""

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from qednet.models.base import BaseTransformer


class StandardScaler(BaseTransformer):
    """Standard scaler wrapper."""

    def __init__(self):
        self._scaler = SklearnStandardScaler()

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "StandardScaler":
        self._scaler.fit(X)
        return self

    def transform(self, X: NDArray) -> NDArray:
        return self._scaler.transform(X)


class RobustScaler(BaseTransformer):
    """Robust scaler wrapper."""

    def __init__(self):
        self._scaler = SklearnRobustScaler()

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "RobustScaler":
        self._scaler.fit(X)
        return self

    def transform(self, X: NDArray) -> NDArray:
        return self._scaler.transform(X)