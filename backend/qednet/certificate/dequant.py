"""Classical dequantization: Random Fourier Features and Nystrom
approximations of the QUANTUM kernel.

Given the QSVM quantum-kernel matrix, we build classical feature
approximations and train the same SVM on them. If a classical approximation
matches the quantum kernel model, the quantum kernel offers no measurable
kernel-space advantage on this dataset (honest dequantization analysis).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


def _rbf_gamma(K: np.ndarray) -> float:
    """Median heuristic gamma from a kernel matrix's nonzero distances."""
    n = K.shape[0]
    if n < 4:
        return 1.0 / max(1, K.shape[1])
    sq = np.clip(2.0 - 2.0 * K, 0, None)  # squared distances from fidelities
    iu = np.triu_indices(n, k=1)
    med = np.median(sq[iu][sq[iu] > 1e-12])
    return float(1.0 / med) if med > 1e-12 else 1.0


def random_fourier_approximation(K_train: np.ndarray, y_train: np.ndarray,
                                 K_test: np.ndarray, y_test: np.ndarray,
                                 n_components: int = 200,
                                 seed: int = 42) -> Dict[str, Any]:
    """RFF approximation of the quantum kernel's implicit feature space.

    The quantum fidelity kernel is positive semi-definite; we approximate a
    shifted RBF surrogate fitted to the quantum kernel's induced distances
    and evaluate whether the classical approximation reproduces performance.
    """
    gamma = _rbf_gamma(K_train)
    rng = np.random.default_rng(seed)
    d = K_train.shape[1]
    W = rng.normal(0, np.sqrt(2 * gamma), size=(d, n_components))
    b = rng.uniform(0, 2 * np.pi, size=n_components)

    def feats(Km):
        # pseudo-features from the kernel rows (Nyström-free RFF surrogate)
        D = np.sqrt(np.clip(2.0 - 2.0 * Km, 0, None)) / np.sqrt(2)
        Z = np.sqrt(2.0 / n_components) * np.cos(D @ W + b)
        return Z

    Z_tr, Z_te = feats(K_train), feats(K_test)
    clf = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
              random_state=seed).fit(Z_tr, y_train)
    p = clf.predict_proba(Z_te)[:, 1]
    auroc = float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) == 2 else None
    return {
        "method": "Random Fourier Features (surrogate of quantum kernel distances)",
        "n_components": n_components, "gamma_median_heuristic": round(gamma, 6),
        "auroc": auroc,
        "note": "classical kernel approximation evaluated on identical test folds",
    }


def nystrom_approximation(K_train: np.ndarray, y_train: np.ndarray,
                          K_test: np.ndarray, y_test: np.ndarray,
                          n_components: int = 100,
                          seed: int = 42) -> Dict[str, Any]:
    """Nystrom low-rank approximation of the quantum kernel matrix."""
    n = K_train.shape[0]
    n_comp = int(min(n_components, n - 1))
    idx = np.random.default_rng(seed).choice(n, size=n_comp, replace=False)
    C = K_train[:, idx]  # (n, m)
    W = K_train[np.ix_(idx, idx)]  # (m, m)
    # SVD-based features: F = C @ B with K ~ F F^T, B = V sqrt(S^-1)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    tol = max(W.shape) * np.finfo(float).eps * (S[0] if len(S) else 0)
    S_inv = np.where(S > tol, 1.0 / np.maximum(S, tol), 0.0)
    B = Vt.T @ np.diag(np.sqrt(S_inv))  # (m, m)
    F_train = C @ B
    F_test = K_test[:, idx] @ B
    clf = SVC(C=1.0, kernel="linear", probability=True,
              random_state=seed).fit(F_train, y_train)
    p = clf.predict_proba(F_test)[:, 1]
    auroc = float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) == 2 else None
    # kernel reconstruction error
    K_hat = F_train @ F_train.T
    err = float(np.linalg.norm(K_train - K_hat) / max(1e-12, np.linalg.norm(K_train)))
    return {
        "method": "Nystrom low-rank approximation of quantum kernel",
        "n_components": n_comp,
        "auroc": auroc,
        "kernel_relative_reconstruction_error": round(err, 6),
        "note": "classical low-rank surrogate evaluated on identical test folds",
    }
