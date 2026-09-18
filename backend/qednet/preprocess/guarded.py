"""Guarded transformer wrappers (F2 leakage enforcement at the object level).

Every learned transformer (imputer, scaler, selector, PCA) is wrapped so that
ANY ``fit``/``fit_transform`` call — from the pipeline, from user code, from
a bug — passes through the ``LeakageGuard``. Fitting while evaluation data
is active raises ``LeakageError``: TEST FAILURE / EXPERIMENT INVALID.
"""
from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin

from ..data.validation import LeakageError, LeakageGuard


class GuardedTransformer(BaseEstimator, TransformerMixin):
    """Delegating transformer whose fit passes the leakage guard."""

    def __init__(self, inner: Any, part: str, guard: LeakageGuard):
        self.inner = inner
        self.part = part
        self.guard = guard

    def fit(self, X, y=None):
        self.guard.record_fit(self.part, len(X))
        self.inner.fit(X, y)
        return self

    def fit_transform(self, X, y=None, **kw):
        self.guard.record_fit(self.part, len(X))
        if hasattr(self.inner, "fit_transform"):
            out = self.inner.fit_transform(X, y, **kw)
        else:
            self.inner.fit(X, y)
            out = self.inner.transform(X)
        self.guard.record_transform(self.part, len(X))
        return out

    def transform(self, X):
        self.guard.record_transform(self.part, len(X))
        return self.inner.transform(X)

    # -- explicit pickle protocol (bypasses BaseEstimator state logic) --
    def __getstate__(self):
        return {"inner": self.inner, "part": self.part, "guard": self.guard}

    def __setstate__(self, state):
        self.inner = state["inner"]
        self.part = state["part"]
        self.guard = state["guard"]

    # -- pass-through of fitted attributes / methods ------------------------
    def __getattr__(self, name: str):
        if name.startswith("__") or name in ("inner", "part", "guard"):
            raise AttributeError(name)
        return getattr(self.inner, name)


class GuardedResampler:
    """Wrap SMOTE-like resamplers: fit_resample only on training folds."""

    def __init__(self, inner: Any, part: str, guard: LeakageGuard):
        self.inner = inner
        self.part = part
        self.guard = guard

    def fit_resample(self, X, y):
        self.guard.record_fit(self.part, len(X))
        return self.inner.fit_resample(X, y)

    def __getstate__(self):
        return {"inner": self.inner, "part": self.part, "guard": self.guard}

    def __setstate__(self, state):
        self.inner = state["inner"]
        self.part = state["part"]
        self.guard = state["guard"]

    def __getattr__(self, name: str):
        if name.startswith("__") or name in ("inner", "part", "guard"):
            raise AttributeError(name)
        return getattr(self.inner, name)
