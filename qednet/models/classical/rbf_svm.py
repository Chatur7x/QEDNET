"""RBF SVM baseline (D6)."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class RBFSVMModel(BaseEstimator, ClassifierMixin):
    """RBF Kernel SVM with sensible defaults."""

    def __init__(
        self,
        C: float = 1.0,
        gamma: str | float = "scale",
        class_weight: str = "balanced",
        probability: bool = True,
        random_state: int = 42,
    ):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        self._model = SVC(
            C=self.C,
            kernel="rbf",
            gamma=self.gamma,
            class_weight=self.class_weight,
            probability=self.probability,
            random_state=self.random_state,
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
        n_sv = self._model.n_support_.sum() if self._model else 0
        return {
            "model": "rbf_svm",
            "C": self.C,
            "gamma": self.gamma,
            "n_support_vectors": int(n_sv),
        }