"""B. Dequantization support (D8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass
class DequantizationResult:
    quantum_kernel: np.ndarray
    rff_kernel: np.ndarray | None
    nystrom_kernel: np.ndarray | None
    rff_error: float | None
    nystrom_error: float | None
    n_features: int
    n_components_rff: int
    n_components_nystrom: int


def _rff_approx(
    K: np.ndarray, n_components: int, gamma: float | None, random_state: int
) -> tuple[np.ndarray, float]:
    rng = np.random.RandomState(random_state)
    n = K.shape[0]
    if gamma is None:
        off_diag = K[~np.eye(n, dtype=bool)]
        off_diag = off_diag[off_diag > 0]
        gamma = 1.0 / (-np.median(np.log(np.clip(off_diag, 1e-10, 1)))) if len(off_diag) else 1.0

    evals, evecs = eigh(K)
    evals = np.maximum(evals, 0)
    Phi = evecs @ np.diag(np.sqrt(evals))

    if n_components >= n:
        return K, 0.0

    W = rng.randn(n, n_components) * np.sqrt(2 * gamma)
    b = rng.uniform(0, 2 * np.pi, n_components)
    Phi_rff = np.sqrt(2.0 / n_components) * np.cos(Phi @ W + b)
    K_rff = Phi_rff @ Phi_rff.T
    return K_rff, float(np.linalg.norm(K - K_rff, "fro"))


def _nystrom_approx(
    K: np.ndarray, n_components: int, random_state: int
) -> tuple[np.ndarray, float]:
    rng = np.random.RandomState(random_state)
    n = K.shape[0]
    n_components = min(n_components, n)
    idx = np.sort(rng.choice(n, n_components, replace=False))
    C = K[:, idx]
    W = K[np.ix_(idx, idx)]
    evals, evecs = eigh(W)
    evals = np.maximum(evals, 1e-10)
    W_inv = evecs @ np.diag(1.0 / evals) @ evecs.T
    K_nys = C @ W_inv @ C.T
    return K_nys, float(np.linalg.norm(K - K_nys, "fro"))


def dequantize_kernel(
    K_quantum: np.ndarray,
    n_rff: int = 100,
    n_nystrom: int = 50,
    gamma: float | None = None,
    random_state: int = 0,
) -> DequantizationResult:
    K_rff, rff_err = _rff_approx(K_quantum, n_rff, gamma, random_state)
    K_nys, nys_err = _nystrom_approx(K_quantum, n_nystrom, random_state)
    return DequantizationResult(
        quantum_kernel=K_quantum,
        rff_kernel=K_rff,
        nystrom_kernel=K_nys,
        rff_error=rff_err,
        nystrom_error=nys_err,
        n_features=K_quantum.shape[0],
        n_components_rff=n_rff,
        n_components_nystrom=n_nystrom,
    )
