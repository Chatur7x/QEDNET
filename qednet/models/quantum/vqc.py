"""VQC: angle/ZZ/amplitude encoding + strongly-entangling layers (D5).

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

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qednet.encoding import validate_matrix
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


class VQC:
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
    ):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if optimizer.lower() not in ("adam", "sgd", "spsa"):
            raise ValueError("optimizer must be adam/sgd/spsa")
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.encoding = encoding.lower()
        self.device_name = device
        self.shots = shots
        self.diff_method_req = diff_method
        self.optimizer = optimizer.lower()
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.use_entanglement = bool(use_entanglement)
        self._built = False

    @property
    def n_params(self) -> int:
        return self.n_layers * self.n_qubits * 3

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

    def fit(self, X, y):
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
        return self

    def decision_scores(self, X):
        if not hasattr(self, "theta_"):
            raise RuntimeError("VQC is not fitted")
        Xm = validate_matrix(X, self.n_qubits, self.encoding)
        return sigmoid(self._forward(Xm, self.theta_))

    def predict_proba(self, X):
        p1 = self.decision_scores(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.decision_scores(X) >= 0.5).astype(int)

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
