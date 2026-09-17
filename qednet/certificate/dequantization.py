"""B. Dequantization support (D8).

Provides quantum-kernel outputs compatible with:
- Random Fourier Features (RFF)
- Nyström approximation

Returns measurable comparison data; does NOT make scientific judgement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel


@dataclass
class DequantizationResult:
    """Result of dequantization comparison."""
    quantum_kernel: np.ndarray
    rff_kernel: np.ndarray | None
    nystrom_kernel: np.ndarray | None
    rff_error: float | None
    nystrom_error: float | None
    n_features: int
    n_components_rff: int
    n_components_nystrom: int

    def __str__(self):
        return (
            f"DequantizationResult(rff_err={self.rff_error}, "
            f"nys_err={self.nystrom_error})"
        )


def kernel_to_rff(
    K_quantum: np.ndarray,
    n_components: int = 100,
    gamma: float | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, float]:
    """Approximate quantum kernel with Random Fourier Features.

    Args:
        K_quantum: (n, n) quantum kernel matrix
        n_components: number of RFF features
        gamma: RBF kernel bandwidth (if None, use median heuristic)
        random_state: seed

    Returns:
        (K_rff, frobenius_error)
    """
    rng = np.random.RandomState(random_state)
    n = K_quantum.shape[0]

    # Estimate gamma from kernel diagonal/off-diagonal if not provided
    if gamma is None:
        # Median heuristic on pairwise distances implied by kernel
        # K(x,y) = exp(-gamma * ||x-y||^2) => ||x-y||^2 = -log(K)/gamma
        # Use off-diagonal median
        off_diag = K_quantum[~np.eye(n, dtype=bool)]
        off_diag = off_diag[off_diag > 0]
        if len(off_diag) > 0:
            gamma = 1.0 / (-np.median(np.log(np.clip(off_diag, 1e-10, 1))))
        else:
            gamma = 1.0

    # Sample random frequencies
    # For RBF kernel: w ~ N(0, 2*gamma*I)
    # RFF: phi(x) = sqrt(2/D) * cos(W x + b)
    # Since we only have K, not X, we approximate by eigendecomposition
    # K = Phi Phi^T => Phi = U sqrt(Lambda)
    evals, evecs = eigh(K_quantum)
    evals = np.maximum(evals, 0)
    Phi = evecs @ np.diag(np.sqrt(evals))

    # Project to n_components random features
    if n_components >= n:
        return K_quantum, 0.0

    W = rng.randn(n, n_components) * np.sqrt(2 * gamma)
    b = rng.uniform(0, 2 * np.pi, n_components)
    Phi_rff = np.sqrt(2.0 / n_components) * np.cos(Phi @ W + b)
    K_rff = Phi_rff @ Phi_rff.T

    error = float(np.linalg.norm(K_quantum - K_rff, "fro"))
    return K_rff, error


def kernel_to_nystrom(
    K_quantum: np.ndarray,
    n_components: int = 50,
    random_state: int = 0,
) -> tuple[np.ndarray, float]:
    """Nyström approximation of quantum kernel.

    Args:
        K_quantum: (n, n) quantum kernel matrix
        n_components: number of landmark points
        random_state: seed

    Returns:
        (K_nystrom, frobenius_error)
    """
    rng = np.random.RandomState(random_state)
    n = K_quantum.shape[0]
    n_components = min(n_components, n)

    # Sample landmark indices
    idx = rng.choice(n, n_components, replace=False)
    idx = np.sort(idx)

    # Nyström: K ≈ C W^{-1} C^T
    C = K_quantum[:, idx]
    W = K_quantum[np.ix_(idx, idx)]

    # Regularize W for stability
    evals, evecs = eigh(W)
    evals = np.maximum(evals, 1e-10)
    W_inv = evecs @ np.diag(1.0 / evals) @ evecs.T

    K_nys = C @ W_inv @ C.T
    error = float(np.linalg.norm(K_quantum - K_nys, "fro"))
    return K_nys, error


def dequantize_kernel(
    K_quantum: np.ndarray,
    n_rff: int = 100,
    n_nystrom: int = 50,
    gamma: float | None = None,
    random_state: int = 0,
) -> DequantizationResult:
    """Run both RFF and Nyström dequantization on a quantum kernel."""
    K_rff, rff_err = kernel_to_rff(K_quantum, n_rff, gamma, random_state)
    K_nys, nys_err = kernel_to_nystrom(K_quantum, n_nystrom, random_state)

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