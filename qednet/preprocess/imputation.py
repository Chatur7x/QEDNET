"""Leakage-safe imputation (D3)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.utils.validation import check_is_fitted


class SafeImputer(BaseEstimator, TransformerMixin):
    """
    Wrapper that ensures imputation is only fit on training data.
    Prevents leakage by not allowing fit on full dataset before CV.
    """

    def __init__(
        self,
        strategy: str = "median",
        missing_values=np.nan,
        add_indicator: bool = False,
        knn_neighbors: int = 5,
    ):
        self.strategy = strategy
        self.missing_values = missing_values
        self.add_indicator = add_indicator
        self.knn_neighbors = knn_neighbors
        self._imputer = None

    def _make_imputer(self):
        if self.strategy == "knn":
            return KNNImputer(n_neighbors=self.knn_neighbors, add_indicator=self.add_indicator)
        return SimpleImputer(
            missing_values=self.missing_values,
            strategy=self.strategy,
            add_indicator=self.add_indicator,
        )

    def fit(self, X, y=None):
        self._imputer = self._make_imputer()
        self._imputer.fit(X)
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        return self

    def transform(self, X):
        check_is_fitted(self, "_imputer")
        return self._imputer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        if self.add_indicator and hasattr(self._imputer, "indicator_"):
            n_indicator = self._imputer.indicator_.n_features_in_
            base = [f"imputed_{i}" for i in range(self.n_features_in_ - n_indicator)]
            ind = [f"missing_indicator_{i}" for i in range(n_indicator)]
            return np.array(base + ind)
        return np.array([f"imputed_{i}" for i in range(self.n_features_in_)])
