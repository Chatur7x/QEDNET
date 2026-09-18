"""QSVM: Quantum kernel classifier using sklearn SVC(precomputed) (D5) - Member A's implementation.

Kernel: K(x,y) = |<phi(x)|phi(y)>|^2 via swap test or statevector overlap.
Exposes kernel matrix caching, symmetry/diagonal tests, and circuit metadata.
O(n^2) cost preserved and exposed per spec.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pennylane as qml
from sklearn.svm import SVC
from scipy.special import expit

from qednet.encoding import validate_vector, zz_feature_map
from qednet.models.base import BaseEstimator
from .base import validate_xy


def _overlap_statevector(dev, encoding_fn, x, y, n_qubits):
    """Compute |<phi(x)|phi(y)>|^2 using statevector overlap (exact, sim only)."""
    @qml.qnode(dev)
    def circ(a, b):
        encoding_fn(a, wires=range(n_qubits))
        qml.adjoint(encoding_fn)(b, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))
    # overlap = |<0| U(x)^dagger U(y) |0>|^2 = prob of all zeros
    probs = circ(x, y)
    return float(probs[0])


def _overlap_swap_test(dev, encoding_fn, x, y, n_qubits):
    """Compute |<phi(x)|phi(y)>|^2 via swap test (hardware compatible)."""
    # ancilla + 2 x n_qubits registers
    total_wires = 1 + 2 * n_qubits
    anc = 0
    reg1 = list(range(1, 1 + n_qubits))
    reg2 = list(range(1 + n_qubits, 1 + 2 * n_qubits))

    @qml.qnode(dev)
    def circ(a, b):
        # prepare |phi(x)> on reg1, |phi(y)> on reg2
        encoding_fn(a, wires=reg1)
        encoding_fn(b, wires=reg2)
        # swap test
        qml.Hadamard(wires=anc)
        for i in range(n_qubits):
            qml.CSWAP(wires=[anc, reg1[i], reg2[i]])
        qml.Hadamard(wires=anc)
        return qml.probs(wires=anc)
    # P(ancilla=0) = (1 + |<x|y>|^2) / 2
    probs = circ(x, y)
    return float(2 * probs[0] - 1)


class QSVM(BaseEstimator):
    """Quantum kernel classifier using sklearn SVC(precomputed).

    Args:
        n_qubits: Number of qubits
        encoding: Feature encoding ('zz', 'iqp', 'angle')
        device: PennyLane device name
        shots: Number of shots (None for exact)
        C: SVM regularization parameter
        kernel_cache_size: sklearn SVC cache size
        use_swap_test: Use swap test (hardware-compatible) vs statevector overlap
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        n_qubits: int = 4,
        encoding: str = "zz",
        device: str = "default.qubit",
        shots: int | None = None,
        C: float = 1.0,
        kernel_cache_size: int = 200,
        use_swap_test: bool = False,
        random_state: int = 42,
        # Config aliases
        **kwargs,
    ):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self._raw_params = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        self.n_qubits = int(n_qubits)
        self.encoding = encoding.lower()
        self.device_name = device
        self.shots = shots
        self.C = float(C)
        self.kernel_cache_size = int(kernel_cache_size)
        self.use_swap_test = bool(use_swap_test)
        self.random_state = int(random_state)
        self._dev = None
        self._encoding_fn = None
        self._kernel_train = None
        self._X_train = None
        self._svc = None
        self.kernel_evals_ = 0
        self.fit_time_s_ = None
        self._fitted = False

    @property
    def n_params(self) -> int:
        return 0  # QSVM has no trainable params; kernel is fixed

    def _get_encoding_fn(self):
        if self.encoding in ("zz", "iqp"):
            return lambda x, wires: zz_feature_map(x, wires=wires)
        elif self.encoding == "angle":
            from qednet.encoding import angle_encoding
            return angle_encoding
        else:
            raise ValueError(f"QSVM: unsupported encoding {self.encoding!r}")

    def _build_device(self):
        if self._dev is not None:
            return
        kwargs = {"wires": self._n_wires()}
        if self.shots is not None:
            kwargs["shots"] = self.shots
        try:
            self._dev = qml.device(self.device_name, **kwargs)
        except Exception:
            self._dev = qml.device("default.qubit", **kwargs)
            self.device_name = "default.qubit"
        self._encoding_fn = self._get_encoding_fn()

    def _n_wires(self) -> int:
        return 1 + 2 * self.n_qubits if self.use_swap_test else self.n_qubits

    def _kernel_element(self, x, y):
        self._build_device()
        self.kernel_evals_ += 1
        if self.use_swap_test:
            return _overlap_swap_test(self._dev, self._encoding_fn, x, y, self.n_qubits)
        return _overlap_statevector(self._dev, self._encoding_fn, x, y, self.n_qubits)

    def _compute_kernel(self, X, Y=None):
        """Compute kernel matrix K(X, Y). Y defaults to X (symmetric)."""
        self._build_device()
        X = np.asarray(X)
        Y = X if Y is None else np.asarray(Y)
        n, m = X.shape[0], Y.shape[0]
        K = np.zeros((n, m), dtype=float)
        t0 = time.time()
        for i in range(n):
            for j in range(m):
                # exploit symmetry when Y is X
                if Y is X and j < i:
                    K[i, j] = K[j, i]
                else:
                    K[i, j] = self._kernel_element(X[i], Y[j])
        return K

    def fit(self, X: NDArray, y: NDArray) -> "QSVM":
        Xm, ym = validate_xy(X, y, self.n_qubits, self.encoding)
        t0 = time.time()
        self._X_train = Xm
        self.kernel_evals_ = 0
        K = self._compute_kernel(Xm)
        self._kernel_train = K
        self._svc = SVC(C=self.C, kernel="precomputed", cache_size=self.kernel_cache_size)
        self._svc.fit(K, ym)
        self.fit_time_s_ = time.time() - t0
        self.n_features_in_ = Xm.shape[1]
        self._fitted = True
        return self

    def _predict_kernel(self, X: NDArray) -> NDArray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        if X_arr.shape[0] == 0:
            raise ValueError("X has no rows")
        validate_vector(X_arr[0], self.n_qubits, self.encoding)
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("features must be finite")
        if self._X_train is None or self._svc is None:
            raise RuntimeError("QSVM not fitted")
        K = self._compute_kernel(X_arr, self._X_train)
        return self._svc.decision_function(K)

    def decision_scores(self, X: NDArray) -> NDArray:
        return self._predict_kernel(X)

    def predict_proba(self, X: NDArray) -> NDArray:
        scores = self.decision_scores(X)
        p1 = expit(scores)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: NDArray) -> NDArray:
        scores = self.decision_scores(X)
        return (scores >= 0).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        # Raw ctor values: sklearn clone() requires identical objects back.
        if hasattr(self, "_raw_params"):
            return dict(self._raw_params)
        return {
            "n_qubits": self.n_qubits,
            "encoding": self.encoding,
            "shots": self.shots,
            "random_state": self.random_state,
            "C": self.C,
            "kernel_cache_size": self.kernel_cache_size,
            "use_swap_test": self.use_swap_test,
            "device": self.device_name,
        }

    def set_params(self, **params) -> "QSVM":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def metadata(self) -> dict:
        enc = self.encoding
        nq = self.n_qubits
        if enc in ("zz", "iqp"):
            depth = 3  # H, RZ, CNOT-RZ-CNOT
            two_q = max(0, nq - 1)
        elif enc == "angle":
            depth = 1
            two_q = 0
        else:
            depth = 1
            two_q = 0
        return {
            "model": "qsvm",
            "n_qubits": nq,
            "encoding": enc,
            "use_swap_test": self.use_swap_test,
            "C": self.C,
            "kernel_evals": self.kernel_evals_,
            "circuit_depth": depth,
            "two_qubit_gates": two_q,
            "shots": self.shots,
            "device": self.device_name,
            "fit_time_s": self.fit_time_s_,
        }