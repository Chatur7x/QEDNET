"""MLP classifier wrapper."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from sklearn.neural_network import MLPClassifier as SklearnMLP
from qednet.models.base import BaseEstimator


class MLPClassifier(BaseEstimator):
    """Multi-layer Perceptron classifier with sklearn-compatible interface."""

    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: int = 42,
        **kwargs,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None

    def fit(self, X: NDArray, y: NDArray) -> "MLPClassifier":
        self._model = SklearnMLP(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
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
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
        params.update(self.kwargs)
        return params