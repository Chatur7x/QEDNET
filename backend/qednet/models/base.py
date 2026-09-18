"""Model interface contract.

Every model — quantum or classical — exposes exactly:
    fit(X, y) / predict(X) / predict_proba(X)
plus honest metadata (parameter count, training/inference time, quantum
resources when applicable). The evaluation engine operates on this contract
alone, with zero model-specific evaluation code.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np


class BaseModel:
    """Common model abstraction for the benchmarking engine."""

    name: str = "base"
    family: str = "unknown"  # "quantum" | "classical"
    supports_predict_proba: bool = True

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = int(random_state)
        self.train_time_s: float = 0.0
        self.inference_time_s: float = 0.0
        self.n_train_samples: int = 0
        self.classes_: Optional[np.ndarray] = None

    # -- contract ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        t0 = time.perf_counter()
        self._fit_impl(np.asarray(X, dtype=np.float64), np.asarray(y))
        self.classes_ = np.array([0, 1])
        self.train_time_s = time.perf_counter() - t0
        self.n_train_samples = int(len(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=np.float64)
        p1 = self._positive_proba(X)
        self.inference_time_s = time.perf_counter() - t0
        return np.column_stack([1.0 - p1, p1])

    # -- to implement ---------------------------------------------------------
    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def _positive_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- metadata ---------------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "family": self.family,
            "n_parameters": self.n_parameters(),
            "train_time_s": self.train_time_s,
            "inference_time_s": self.inference_time_s,
            "n_train_samples": self.n_train_samples,
        }

    def n_parameters(self) -> int:
        """Total trainable parameter count."""
        raise NotImplementedError

    def quantum_resources(self) -> Optional[Dict[str, Any]]:
        """Quantum resource report; None for classical models."""
        return None


def positive_class_index(y: np.ndarray) -> int:
    """Convention: label 1 is the positive (disease-risk) class."""
    return 1
