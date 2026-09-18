"""VQC: Variational Quantum Classifier (D5) - Member A's implementation.

Architecture: input -> encoding -> L x [Rot(3 params/qubit) + CNOT chain]
-> <Z0> -> sigmoid -> BCE. Fits sklearn-style fit/predict_proba/predict.

Gradient modes: backprop (default.qubit) / adjoint (lightning.qubit),
parameter-shift (hardware-compatible), finite-diff. Optimizers: adam/sgd/spsa.
SPSA uses 2 circuit-batch evaluations per step, robust under shot noise.

Set use_entanglement=False for the parameter-matched ablation: entangling
CNOTs are removed but the parameter tensor shape (L, n_qubits, 3) is kept,
so capacity is held constant and any gap isolates entanglement.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pennylane as qml
from pennylane import numpy as pnp

from qednet.encoding import validate_vector
from qednet.models.base import BaseEstimator
from .base import bce_loss, sigmoid, validate_xy


def _resolve_diff_method(requested: str, device_name: str) -> str:
    r = requested.lower().replace("_", "-")
    if r in ("backprop", "back-prop"):
        return "adjoint" if "lightning" in device_name else "backprop"
    if r in ("parameter-shift", "parameter-shift-rule", "ps"):
        return "parameter-shift"
    if r in ("finite-diff", "finite-difference", "spsa-grad"):
        return "finite-diff"
    raise ValueError(f"unknown diff_method {requested!r}")


class VQC(BaseEstimator):
    """Variational Quantum Classifier with sklearn-compatible interface.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        encoding: Feature encoding ('angle', 'zz', 'amplitude')
        device: PennyLane device name
        shots: Number of shots (None for exact)
        diff_method: Gradient method ('backprop', 'parameter-shift', 'finite-diff')
        optimizer: Optimizer ('adam', 'sgd', 'spsa')
        lr: Learning rate
        max_iter: Maximum iterations
        seed: Random seed
        use_entanglement: Whether to use entangling CNOTs
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: str = "angle",
        device: str = "default.qubit",
        shots: int | None = None,
        diff_method: str = "backprop",
        optimizer: str = "adam",
        lr: float = 0.05,
        max_iter: int = 30,
        seed: int = 0,
        use_entanglement: bool = True,
        # Aliases for config compatibility
        layers: int | None = None,
        ansatz: str | None = None,
        learning_rate: float | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        # Raw ctor values for sklearn clone() (identity check needs same objects)
        self._raw_params = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        # Handle config aliases
        self.n_qubits = int(n_qubits)
        self.n_layers = int(layers if layers is not None else n_layers)
        self.encoding = encoding.lower()
        self.device_name = device
        self.shots = shots
        self.diff_method_req = diff_method
        self.optimizer = optimizer.lower()
        self.lr = float(learning_rate if learning_rate is not None else lr)
        self.max_iter = int(max_iter)
        self.seed = int(random_state if random_state is not None else seed)
        self.use_entanglement = bool(use_entanglement)
        self._built = False

        # Store aliases for get_params
        self._ansatz = ansatz
        self._learning_rate = learning_rate
        self._random_state = random_state
        self._layers = layers

    @property
    def n_params(self) -> int:
        return self.n_layers * self.n_qubits * 3

    @property
    def layers(self) -> int:
        return self.n_layers

    @layers.setter
    def layers(self, value: int):
        self.n_layers = value

    def _build(self):
        diff = _resolve_diff_method(self.diff_method_req, self.device_name)
        self.diff_method_ = diff
        kwargs = {"wires": self.n_qubits}
        if self.shots is not None:
            kwargs["shots"] = self.shots
        try:
            self._dev = qml.device(self.device_name, **kwargs)
        except Exception:
            self._dev = qml.device("default.qubit", **kwargs)
            self.device_name = "default.qubit"
            self.diff_method_ = _resolve_diff_method(self.diff_method_req, self.device_name)

        enc = self.encoding
        nq, nl = self.n_qubits, self.n_layers
        ent = self.use_entanglement

        @qml.qnode(self._dev, diff_method=self.diff_method_)
        def circuit(x, theta):
            if enc == "angle":
                for i in range(min(len(x), nq)):
                    qml.RX(x[i], wires=i)
            elif enc in ("zz", "iqp"):
                for i in range(nq):
                    qml.Hadamard(wires=i)
                for i in range(min(len(x), nq)):
                    qml.RZ(x[i], wires=i)
                for i in range(min(len(x), nq) - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(x[i] * x[i + 1], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            elif enc == "amplitude":
                from qednet.encoding.base import pad_to_pow2

                qml.StatePrep(pad_to_pow2(pnp.asarray(x, dtype=float), nq), wires=range(nq))
            else:
                raise ValueError(f"VQC: unsupported encoding {enc!r}")
            for layer in range(nl):
                for i in range(nq):
                    qml.Rot(theta[layer, i, 0], theta[layer, i, 1],
                            theta[layer, i, 2], wires=i)
                if ent:
                    for i in range(nq - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if nq > 2:
                        qml.CNOT(wires=[nq - 1, 0])
            return qml.expval(qml.Z(0))

        self._circuit = circuit
        self._built = True

    def _forward(self, X, theta):
        """Return circuit outputs for each x in X as a 1D array/tensor."""
        return pnp.array([self._circuit(x, theta) for x in X])

    def fit(self, X: NDArray, y: NDArray) -> "VQC":
        Xm, ym = validate_xy(X, y, self.n_qubits, self.encoding)
        if not self._built:
            self._build()
        rng = np.random.RandomState(self.seed)
        theta = pnp.array(rng.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits, 3)),
                          requires_grad=True)
        t0 = time.time()

        def batch_loss(t):
            z = self._forward(Xm, t)
            return bce_loss(sigmoid(z), ym)

        losses, grad_vars = [], []
        if self.optimizer in ("adam", "sgd"):
            opt = qml.AdamOptimizer(self.lr) if self.optimizer == "adam" \
                else qml.GradientDescentOptimizer(self.lr)
            gfn = qml.grad(batch_loss)
            for _ in range(self.max_iter):
                theta, prev = opt.step_and_cost(batch_loss, theta)
                g = np.asarray(gfn(theta)).ravel()
                losses.append(float(prev))
                grad_vars.append(float(np.var(g)))
        else:  # SPSA: 2 evaluations/step, no analytic grad
            ck, rng_s = 0.1, np.random.RandomState(self.seed)
            for k in range(self.max_iter):
                delta = rng_s.choice([-1.0, 1.0], size=theta.shape)
                lp = batch_loss(theta + ck * delta)
                lm = batch_loss(theta - ck * delta)
                ghat = (lp - lm) / (2 * ck) * delta
                theta = pnp.array(np.asarray(theta) - self.lr / (k + 1) ** 0.6 * ghat,
                                  requires_grad=True)
                losses.append(float((lp + lm) / 2))
                grad_vars.append(float(np.var(np.asarray(ghat).ravel())))
        self.theta_ = np.asarray(theta)
        self.train_loss_ = losses
        self.grad_variances_ = grad_vars
        self.n_iter_ = self.max_iter
        self.fit_time_s_ = time.time() - t0
        self.n_features_in_ = Xm.shape[1]
        self._fitted = True
        return self

    def decision_scores(self, X: NDArray) -> NDArray:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("VQC is not fitted")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        if X_arr.shape[0] == 0:
            raise ValueError("X has no rows")
        validate_vector(X_arr[0], self.n_qubits, self.encoding)
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("features must be finite")
        return sigmoid(self._forward(X_arr, self.theta_))

    def predict_proba(self, X: NDArray) -> NDArray:
        p1 = self.decision_scores(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: NDArray) -> NDArray:
        return (self.decision_scores(X) >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        # Raw ctor values: sklearn clone() requires identical objects back.
        if hasattr(self, "_raw_params"):
            return dict(self._raw_params)
        return {
            "n_qubits": self.n_qubits,
            "layers": self.n_layers,
            "n_layers": self.n_layers,
            "ansatz": self._ansatz or "strongly_entangling",
            "encoding": self.encoding,
            "optimizer": self.optimizer,
            "learning_rate": self._learning_rate or self.lr,
            "max_iter": self.max_iter,
            "shots": self.shots,
            "random_state": self._random_state or self.seed,
            "use_entanglement": self.use_entanglement,
            "device": self.device_name,
            "diff_method": getattr(self, "diff_method_", self.diff_method_req),
        }

    def set_params(self, **params) -> "VQC":
        for key, value in params.items():
            if key == "layers":
                self.n_layers = value
            elif key == "learning_rate":
                self.lr = value
                self._learning_rate = value
            elif key == "random_state":
                self.seed = value
                self._random_state = value
            elif hasattr(self, key):
                setattr(self, key, value)
        return self

    def metadata(self) -> dict:
        two_q = (self.n_qubits if self.n_qubits > 1 else 0) * self.n_layers \
            if self.use_entanglement else 0
        return {
            "model": "vqc",
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "encoding": self.encoding,
            "n_params": self.n_params,
            "two_qubit_gates": two_q,
            "shots": self.shots,
            "diff_method": getattr(self, "diff_method_", self.diff_method_req),
            "device": self.device_name,
            "use_entanglement": self.use_entanglement,
            "fit_time_s": getattr(self, "fit_time_s_", None),
        }


def _resolve_diff_method(requested: str, device_name: str) -> str:
    r = requested.lower().replace("_", "-")
    if r in ("backprop", "back-prop"):
        return "adjoint" if "lightning" in device_name else "backprop"
    if r in ("parameter-shift", "parameter-shift-rule", "ps"):
        return "parameter-shift"
    if r in ("finite-diff", "finite-difference", "spsa-grad"):
        return "finite-diff"
    raise ValueError(f"unknown diff_method {requested!r}")