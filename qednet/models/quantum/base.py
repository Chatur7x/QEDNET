"""Shared validation + helpers for quantum estimators (D5)."""

from __future__ import annotations

import numpy as np
from pennylane import numpy as pnp


def validate_xy(X, y, n_qubits: int, encoding: str):
    from qednet.encoding import validate_matrix

    Xm = validate_matrix(X, n_qubits, encoding)
    arr = np.asarray(y).ravel()
    if arr.shape[0] != Xm.shape[0]:
        raise ValueError(f"X/y size mismatch: {Xm.shape[0]} vs {arr.shape[0]}")
    uniq = set(np.unique(arr).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"y must be binary 0/1, got {sorted(uniq)}")
    return Xm, arr.astype(int)


def sigmoid(z):
    """Autograd-compatible sigmoid."""
    return 1.0 / (1.0 + pnp.exp(-z))


def bce_loss(p, y, eps: float = 1e-7):
    p = pnp.clip(p, eps, 1 - eps)
    y = pnp.asarray(y)
    return -pnp.mean(y * pnp.log(p) + (1 - y) * pnp.log(1 - p))
