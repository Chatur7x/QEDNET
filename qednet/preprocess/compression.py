"""Feature compression and selection."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from qednet.models.base import BaseTransformer


class PCACompressor(BaseTransformer):
    """PCA-based feature compression."""

    def __init__(self, n_components: int = 8, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "PCACompressor":
        self._pca.fit(X)
        return self

    def transform(self, X: NDArray) -> NDArray:
        return self._pca.transform(X)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self._pca.explained_variance_ratio_


class FeatureSelector(BaseTransformer):
    """Univariate feature selection."""

    def __init__(self, k: int = 8, score_func=f_classif):
        self.k = k
        self.score_func = score_func
        self._selector = SelectKBest(score_func=score_func, k=k)

    def fit(self, X: NDArray, y: NDArray) -> "FeatureSelector":
        self._selector.fit(X, y)
        return self

    def transform(self, X: NDArray) -> NDArray:
        return self._selector.transform(X)

    def get_support(self) -> np.ndarray:
        return self._selector.get_support()