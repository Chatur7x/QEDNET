"""Shared training utilities for variational quantum models.

Adam optimiser over autograd-tracked parameter dicts plus a fixed quantum
input conditioning transform (standardised -> [0, 1] via logistic squash).

The squash is a FIXED (non-learned, data-independent) transform, so it cannot
leak information across folds. Both quantum models and the Bottleneck
Honesty Meter use it, guaranteeing the honesty comparison sees exactly the
representation the quantum layer receives.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import autograd.numpy as anp
import numpy as np


def quantum_input_transform(X: np.ndarray) -> np.ndarray:
    """Fixed logistic squash: standardised features -> [0, 1].

    Applied before angle-style quantum encoding. Non-learned and
    data-independent (no fit step), hence leakage-free by construction.
    """
    X = anp.asarray(X, dtype=np.float64)
    return 0.5 * (1.0 + anp.tanh(0.5 * X))


class Adam:
    """Minimal Adam optimiser for autograd parameter dicts."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 0.05,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.t = 0
        self.m = {k: np.zeros_like(np.asarray(v)) for k, v in params.items()}
        self.v = {k: np.zeros_like(np.asarray(v)) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray],
             grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.t += 1
        out = {}
        for k, g in grads.items():
            g = np.asarray(g)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            out[k] = np.asarray(params[k]) - self.lr * mhat / (
                np.sqrt(vhat) + self.eps)
        return out

    @property
    def total_params(self) -> int:
        return int(sum(np.asarray(v).size for v in self.m.values()))


def bce_loss(logits: anp.ndarray, y: anp.ndarray) -> anp.ndarray:
    """Numerically stable binary cross-entropy on logits."""
    return -anp.mean(y * anp.log(logits + 1e-10) +
                     (1 - y) * anp.log(1 - logits + 1e-10))


def sigmoid(x: anp.ndarray) -> anp.ndarray:
    """Overflow-safe sigmoid via tanh (autograd-friendly, no branching)."""
    return 0.5 * (1.0 + anp.tanh(0.5 * x))


def count_params(params: Dict[str, np.ndarray]) -> int:
    return int(sum(np.asarray(v).size for v in params.values()))


def batches(n: int, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    """Shuffled index batches."""
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]
