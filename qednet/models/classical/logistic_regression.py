"""Logistic Regression baseline (D6)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class LogisticRegressionModel(BaseEstimator, ClassifierMixin):
    """Logistic Regression with sensible defaults for biomedical data."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model = None

    def fit(self, X, y):
        self._model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_model")
        return self._model.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, "_model")
        return self._model.predict(X)

    def decision_scores(self, X):
        check_is_fitted(self, "_model")
        return self._model.decision_function(X)

    def metadata(self) -> dict:
        return {
            "model": "logistic_regression",
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "n_params": self._model.coef_.size + self._model.intercept_.size if self._model else 0,
        }