"""Leakage-safe feature selection (D3)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.utils.validation import check_is_fitted


class SafeFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Leakage-safe feature selection. Only fits on training data.
    Supports variance threshold, mutual information, and F-statistic.
    """

    def __init__(
        self,
        method: str = "mutual_info",
        k: int = 16,
        variance_threshold: float = 0.0,
        random_state: int = 42,
    ):
        self.method = method.lower()
        self.k = k
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self._selector = None
        self._var_filter = None

    def _make_selector(self):
        if self.method == "f_classif":
            return SelectKBest(f_classif, k=self.k)
        if self.method == "variance":
            return VarianceThreshold(threshold=self.variance_threshold)
        return SelectKBest(mutual_info_classif, k=self.k)

    def fit(self, X, y):
        # First remove zero-variance features
        self._var_filter = VarianceThreshold(threshold=self.variance_threshold)
        X_var = self._var_filter.fit_transform(X)

        if X_var.shape[1] == 0:
            raise ValueError("All features removed by variance threshold")

        self._selector = self._make_selector()
        if self.method != "variance":
            self._selector.fit(X_var, y)
        self.n_features_in_ = X.shape[1]
        self.selected_features_ = self._var_filter.get_support(indices=True)
        if self.method != "variance":
            self.selected_features_ = self.selected_features_[
                self._selector.get_support(indices=True)
            ]
        return self

    def transform(self, X):
        check_is_fitted(self, "_selector")
        X_var = self._var_filter.transform(X)
        if self.method != "variance":
            return self._selector.transform(X_var)
        return X_var

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "_selector")
        if input_features is not None:
            input_features = np.array(input_features)
            return input_features[self.selected_features_]
        return np.array([f"selected_{i}" for i in range(len(self.selected_features_))])
