"""Data Re-uploading Classifier (D5).

Repeatedly injects features between trainable layers:
|0> -> U(theta_1) -> S(x) -> U(theta_2) -> S(x) -> ... -> U(theta_L) -> S(x) -> measure Z0

S(x) = product RY(x_i + phi_i) on each qubit (parameterized feature map).
Parameters: {theta_l (L x n_qubits x 3), phi_l (L x n_qubits)}.
Configurable layers, parameter count, deterministic seed.
"""

from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qednet.encoding import validate_vector
from .base import bce_loss, sigmoid, validate_xy


class ReuploadClassifier:
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        device: str = "default.qubit",
        shots: int | None = None,
        diff_method: str = "backprop",
        optimizer: str = "adam",
        lr: float = 0.05,
        max_iter: int = 30,
        seed: int = 0,
    ):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if optimizer.lower() not in ("adam", "sgd", "spsa"):
            raise ValueError("optimizer must be adam/sgd/spsa")
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.device_name = device
        self.shots = shots
        self.diff_method_req = diff_method
        self.optimizer = optimizer.lower()
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self._built = False

    @property
    def n_params(self) -> int:
        # theta: L x n_qubits x 3, phi: L x n_qubits
        return self.n_layers * self.n_qubits * 3 + self.n_layers * self.n_qubits

    def _resolve_diff(self):
        r = self.diff_method_req.lower().replace("_", "-")
        if r in ("backprop", "back-prop"):
            return "adjoint" if "lightning" in self.device_name else "backprop"
        if r in ("parameter-shift", "ps"):
            return "parameter-shift"
        if r in ("finite-diff", "finite-difference"):
            return "finite-diff"
        raise ValueError(f"unknown diff_method {self.diff_method_req!r}")

    def _build(self):
        diff = self._resolve_diff()
        self.diff_method_ = diff
        kwargs = {"wires": self.n_qubits}
        if self.shots is not None:
            kwargs["shots"] = self.shots
        try:
            self._dev = qml.device(self.device_name, **kwargs)
        except Exception:
            self._dev = qml.device("default.qubit", **kwargs)
            self.device_name = "default.qubit"
            self.diff_method_ = self._resolve_diff()

        nq, nl = self.n_qubits, self.n_layers

        @qml.qnode(self._dev, diff_method=self.diff_method_)
        def circuit(x, theta, phi):
            # theta: (nl, nq, 3), phi: (nl, nq)
            for layer in range(nl):
                # trainable layer: Rot on each qubit
                for i in range(nq):
                    qml.Rot(theta[layer, i, 0], theta[layer, i, 1], theta[layer, i, 2], wires=i)
                # data re-uploading layer: RY(x_i + phi_{l,i})
                for i in range(nq):
                    v = float(x[i]) if i < len(x) else 0.0
                    qml.RY(v + phi[layer, i], wires=i)
                # entangling layer
                for i in range(nq - 1):
                    qml.CNOT(wires=[i, i + 1])
                if nq > 2:
                    qml.CNOT(wires=[nq - 1, 0])
            return qml.expval(qml.Z(0))

        self._circuit = circuit
        self._built = True

    def _forward(self, X, theta, phi):
        return pnp.array([self._circuit(x, theta, phi) for x in X])

    def fit(self, X, y):
        Xm, ym = validate_xy(X, y, self.n_qubits, "reupload")
        if not self._built:
            self._build()
        rng = np.random.RandomState(self.seed)
        n_theta = self.n_layers * self.n_qubits * 3
        n_phi = self.n_layers * self.n_qubits
        theta = pnp.array(rng.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits, 3)),
                          requires_grad=True)
        phi = pnp.array(rng.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits)),
                        requires_grad=True)
        t0 = time.time()

        def batch_loss(t, p):
            z = self._forward(Xm, t, p)
            return bce_loss(sigmoid(z), ym)

        losses, grad_vars = [], []
        if self.optimizer in ("adam", "sgd"):
            opt = qml.AdamOptimizer(self.lr) if self.optimizer == "adam" \
                else qml.GradientDescentOptimizer(self.lr)
            gfn = qml.grad(batch_loss, argnums=[0, 1])
            for _ in range(self.max_iter):
                (theta, phi), prev = opt.step_and_cost(batch_loss, theta, phi)
                g_t, g_p = gfn(theta, phi)
                g = pnp.concatenate([pnp.asarray(g_t).ravel(), pnp.asarray(g_p).ravel()])
                losses.append(float(prev))
                grad_vars.append(float(pnp.var(g)))
        else:  # SPSA
            ck, rng_s = 0.1, np.random.RandomState(self.seed)
            for k in range(self.max_iter):
                d_t = rng_s.choice([-1.0, 1.0], size=theta.shape)
                d_p = rng_s.choice([-1.0, 1.0], size=phi.shape)
                lp = batch_loss(theta + ck * d_t, phi + ck * d_p)
                lm = batch_loss(theta - ck * d_t, phi - ck * d_p)
                g_t = (lp - lm) / (2 * ck) * d_t
                g_p = (lp - lm) / (2 * ck) * d_p
                theta = pnp.array(pnp.asarray(theta) - self.lr / (k + 1) ** 0.6 * g_t,
                                  requires_grad=True)
                phi = pnp.array(pnp.asarray(phi) - self.lr / (k + 1) ** 0.6 * g_p,
                                requires_grad=True)
                losses.append(float((lp + lm) / 2))
                grad_vars.append(float(pnp.var(pnp.concatenate([pnp.asarray(g_t).ravel(), pnp.asarray(g_p).ravel()]))))
        self.theta_ = pnp.asarray(theta)
        self.phi_ = pnp.asarray(phi)
        self.train_loss_ = losses
        self.grad_variances_ = grad_vars
        self.n_iter_ = self.max_iter
        self.fit_time_s_ = time.time() - t0
        self.n_features_in_ = Xm.shape[1]
        return self

    def decision_scores(self, X):
        if not hasattr(self, "theta_"):
            raise RuntimeError("ReuploadClassifier not fitted")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        if X_arr.shape[0] == 0:
            raise ValueError("X has no rows")
        validate_vector(X_arr[0], self.n_qubits, "reupload")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("features must be finite")
        z = self._forward(X_arr, self.theta_, self.phi_)
        return sigmoid(z)

    def predict_proba(self, X):
        p1 = self.decision_scores(X)
        return pnp.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.decision_scores(X) >= 0.5).astype(int)

    def metadata(self) -> dict:
        two_q = (self.n_qubits if self.n_qubits > 1 else 0) * self.n_layers
        return {
            "model": "reupload",
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": self.n_params,
            "two_qubit_gates": two_q,
            "shots": self.shots,
            "diff_method": getattr(self, "diff_method_", self.diff_method_req),
            "device": self.device_name,
            "fit_time_s": getattr(self, "fit_time_s_", None),
        }