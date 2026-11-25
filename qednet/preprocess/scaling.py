"""Leakage-safe scaling (D3)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted


class SafeScaler(BaseEstimator, TransformerMixin):
    """
    Wrapper that ensures scaling is only fit on training data.
    Supports standard, minmax, and robust scaling.
    """

    def __init__(self, method: str = "standard", **kwargs):
        self.method = method.lower()
        self.kwargs = kwargs
        self._scaler = None

    def _make_scaler(self):
        if self.method == "minmax":
            return MinMaxScaler(**self.kwargs)
        if self.method == "robust":
            return RobustScaler(**self.kwargs)
        return StandardScaler(**self.kwargs)

    def fit(self, X, y=None):
        self._scaler = self._make_scaler()
        self._scaler.fit(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "_scaler")
        return self._scaler.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array([f"scaled_{i}" for i in range(self.n_features_in_)])

    def inverse_transform(self, X):
        check_is_fitted(self, "_scaler")
        return self._scaler.inverse_transform(X)
