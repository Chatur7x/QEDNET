"""Logistic Regression classifier wrapper."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from sklearn.linear_model import LogisticRegression as SklearnLogistic
from qednet.models.base import BaseEstimator


class LogisticRegressionModel(BaseEstimator):
    """Logistic Regression classifier with sklearn-compatible interface."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        **kwargs,
    ):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None

    def fit(self, X: NDArray, y: NDArray) -> "LogisticRegressionModel":
        self._model = SklearnLogistic(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
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
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
        params.update(self.kwargs)
        return params