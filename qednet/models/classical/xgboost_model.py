"""XGBoost baseline (D6) - strong classical baseline."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class XGBoostModel(BaseEstimator, ClassifierMixin):
    """XGBoost with sensible defaults for tabular biomedical data."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_weight: int = 3,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        verbosity: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self._model = None

    def fit(self, X, y):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise RuntimeError("xgboost not installed")
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            eval_metric="logloss",
            tree_method="hist",
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
        # XGBoost doesn't have decision_function, use log-odds
        proba = self._model.predict_proba(X)
        eps = 1e-15
        proba = np.clip(proba, eps, 1 - eps)
        return np.log(proba[:, 1] / proba[:, 0])

    def metadata(self) -> dict:
        n_trees = self.n_estimators if self._model else 0
        return {
            "model": "xgboost",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_trees": n_trees,
        }