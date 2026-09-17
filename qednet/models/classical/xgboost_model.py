"""XGBoost classifier wrapper."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from qednet.models.base import BaseEstimator

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class XGBoostClassifier(BaseEstimator):
    """XGBoost classifier with sklearn-compatible interface."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = 1,
        **kwargs,
    ):
        if xgb is None:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self._model = None

    def fit(self, X: NDArray, y: NDArray) -> "XGBoostClassifier":
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="logloss",
            **self.kwargs,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        params.update(self.kwargs)
        return params