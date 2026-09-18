"""F6 — Classical Baseline Suite.

Strong, deliberately un-weakened baselines:
  * LogisticRegression — reference linear model
  * RandomForest       — nonlinear tree ensemble
  * XGBoost            — strong tabular biomedical baseline
  * RBF SVM            — classical counterpart to quantum kernels
  * MatchedMLP         — parameter-matched comparator for efficiency studies

Hyperparameter budgets are controlled, configurable and logged.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..base import BaseModel

# Controlled, logged hyperparameter budgets (documented per model).
BUDGETS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "C": 1.0, "max_iter": 2000, "note": "reference linear baseline"},
    "random_forest": {
        "n_estimators": 300, "max_depth": None, "min_samples_leaf": 2,
        "note": "nonlinear tree ensemble baseline"},
    "xgboost": {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.08,
        "subsample": 0.9, "colsample_bytree": 0.9,
        "note": "strong tabular baseline — NOT weakened"},
    "rbf_svm": {
        "C": 1.0, "gamma": "scale", "note": "classical kernel counterpart"},
    "matched_mlp": {
        "note": "hidden layer sized to match a quantum model's parameter "
                "count for parameter-efficiency comparison"},
}


class LogisticRegressionModel(BaseModel):
    name = "logistic_regression"
    family = "classical"

    def __init__(self, random_state: int = 42, C: float = 1.0, **kw):
        super().__init__(random_state)
        self.model = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs", random_state=random_state)

    def _fit_impl(self, X, y):
        self.model.fit(X, y)

    def _positive_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def n_parameters(self) -> int:
        return int(self.model.coef_.size + self.model.intercept_.size)


class RandomForestModel(BaseModel):
    name = "random_forest"
    family = "classical"

    def __init__(self, random_state: int = 42, n_estimators: int = 300, **kw):
        super().__init__(random_state)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1)

    def _fit_impl(self, X, y):
        self.model.fit(X, y)

    def _positive_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def n_parameters(self) -> int:
        # trees as parameter count: total split thresholds (measured post-fit)
        n = 0
        for est in self.model.estimators_:
            n += est.tree_.node_count
        return int(n)


class XGBoostModel(BaseModel):
    name = "xgboost"
    family = "classical"

    def __init__(self, random_state: int = 42, n_estimators: int = 300,
                 max_depth: int = 4, learning_rate: float = 0.08, **kw):
        super().__init__(random_state)
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss",
            random_state=random_state, n_jobs=-1, verbosity=0)

    def _fit_impl(self, X, y):
        self.model.fit(X, y)

    def _positive_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def n_parameters(self) -> int:
        # total tree nodes across boosting rounds (measured post-fit)
        try:
            df = self.model.get_booster().trees_to_dataframe()
            return int(len(df))
        except Exception:  # noqa: BLE001
            try:
                trees = self.model.get_booster().get_dump(dump_format="json")
                return int(sum(t.count('"nodeid":') for t in trees))
            except Exception:  # noqa: BLE001
                return int(self.model.n_estimators * (2 ** self.model.max_depth))


class RBFSVMModel(BaseModel):
    name = "rbf_svm"
    family = "classical"

    def __init__(self, random_state: int = 42, C: float = 1.0, **kw):
        super().__init__(random_state)
        # probability=True enables Platt scaling inside the SVM
        self.model = make_pipeline(
            StandardScaler(),
            SVC(C=C, kernel="rbf", gamma="scale", probability=True,
                random_state=random_state))

    def _fit_impl(self, X, y):
        self.model.fit(X, y)

    def _positive_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def n_parameters(self) -> int:
        svc = self.model.steps[-1][1]
        return int(svc.n_support_.sum())  # support vectors as capacity proxy


class MatchedMLPModel(BaseModel):
    """MLP whose hidden layer is sized to match a target parameter count.

    Used for parameter-efficiency comparisons: given a quantum model with P
    parameters and input dim d, the matched MLP uses
    hidden = max(4, round((P - d - 2) / (d + 2))) so total weights are close
    to P. This keeps the comparison honest (approximate capacity match,
    recorded in metadata).
    """
    name = "matched_mlp"
    family = "classical"

    def __init__(self, random_state: int = 42, match_params: int = 100,
                 input_dim: Optional[int] = None, **kw):
        super().__init__(random_state)
        self.match_params = int(match_params)
        self.hidden = None
        self.model = None

    def _fit_impl(self, X, y):
        d = X.shape[1]
        # total params of 1-hidden MLP: d*h + h + h + 1 (+L2 reg)
        h = max(4, int(round((self.match_params - d - 2) / (d + 2))))
        self.hidden = h
        self.model = MLPClassifier(
            hidden_layer_sizes=(h,), activation="tanh", solver="adam",
            alpha=1e-4, batch_size=min(64, max(8, X.shape[0] // 8)),
            learning_rate_init=1e-2, max_iter=400, early_stopping=True,
            n_iter_no_change=25, random_state=self.random_state)
        self.model.fit(X, y)

    def _positive_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def n_parameters(self) -> int:
        if self.model is None:
            return 0
        w = self.model.coefs_
        b = self.model.intercepts_
        return int(sum(x.size for x in w) + sum(x.size for x in b))

    def metadata(self) -> Dict[str, Any]:
        m = super().metadata()
        m["matched_to_params"] = self.match_params
        m["hidden_units"] = self.hidden
        m["budget_note"] = BUDGETS["matched_mlp"]["note"]
        return m
