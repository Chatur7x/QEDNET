"""Missing value imputer."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Literal
from sklearn.impute import SimpleImputer
from qednet.models.base import BaseTransformer


class MissingValueImputer(BaseTransformer):
    """Missing value imputer wrapper."""

    def __init__(self, strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean", fill_value: float = 0.0):
        self.strategy = strategy
        self.fill_value = fill_value
        self._imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "MissingValueImputer":
        self._imputer.fit(X)
        return self

    def transform(self, X: NDArray) -> NDArray:
        return self._imputer.transform(X)