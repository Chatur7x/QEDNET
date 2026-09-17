"""Random Forest baseline (D6)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class RandomForestModel(BaseEstimator, ClassifierMixin):
    """Random Forest with sensible defaults for biomedical data."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model = None

    def fit(self, X, y):
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
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

    def metadata(self) -> dict:
        n_params = sum(t.tree_.node_count for t in self._model.estimators_) if self._model else 0
        return {
            "model": "random_forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "n_params_approx": n_params,
        }