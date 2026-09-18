"""F5 — Data Re-Uploading Classifier.

Features are re-injected into every trainable quantum layer (Perez-Salinas
et al. 2020). Per layer, per qubit:
    RY(phi[l, q])               (trainable processing rotation)
    RY(w[l, q] * x_q + b[l, q]) (trainable-scaled feature re-injection)
followed by CNOT entanglement (ring). Multi-qubit readout: mean Pauli-Z.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import autograd.numpy as anp
import numpy as np
from autograd import grad

from ...encoding import get_encoder
from ...models.base import BaseModel
from ...models.training import (Adam, bce_loss, batches, count_params,
                                quantum_input_transform, sigmoid)
from ...sim.statevector import (BatchedCircuit, circuit_depth,
                                resource_report)


class DataReuploadingClassifier(BaseModel):
    name = "reupload"
    family = "quantum"

    def __init__(self, random_state: int = 42, n_qubits: int = 8,
                 layers: int = 3, encoding: str = "reupload",
                 backend: str = "numpy", entanglement: bool = True,
                 epochs: int = 60, batch_size: int = 32, lr: float = 0.08,
                 patience: int = 12, shots: Optional[int] = None, **kw):
        super().__init__(random_state)
        if encoding not in ("reupload", "angle"):
            raise ValueError("reupload model uses 'reupload' or 'angle' encoding")
        self.n_qubits = int(n_qubits)
        self.layers = int(layers)
        self.encoding = encoding
        self.backend = backend
        self.entanglement = bool(entanglement)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.patience = int(patience)
        self.shots = shots
        self.params: Dict[str, anp.ndarray] = {}
        self.history: list = []
        self.n_features_seen: int = 0

    def _feature_for_qubit(self, q: int) -> int:
        return q % max(1, self.n_features_seen)

    def _circuit(self, Xsq: anp.ndarray, P: Dict[str, anp.ndarray],
                 gate_counter: Optional[BatchedCircuit] = None):
        B = Xsq.shape[0]
        n_f = Xsq.shape[1]
        cir = BatchedCircuit(self.n_qubits)
        state = cir.zero_state(B)
        ring = [(q, (q + 1) % self.n_qubits) for q in range(self.n_qubits)]
        for l in range(self.layers):
            for q in range(self.n_qubits):
                f = q % n_f
                # processing rotation (trainable)
                state = cir.ry(state, q, P["phi"][l, q])
                # feature re-injection with trainable scale and offset
                ang = P["w"][l, q] * (anp.pi * Xsq[:, f]) + P["b"][l, q]
                state = cir.ry(state, q, ang)
            if self.entanglement and self.n_qubits > 1:
                for c, t in ring:
                    state = cir.cnot(state, c, t)
        out = cir.mean_z(state)  # multi-qubit readout
        if gate_counter is not None:
            gate_counter.gate_counts.update(cir.gate_counts)
        return out

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        Xsq = quantum_input_transform(X)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_seen = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        self.params = {
            "phi": rng.uniform(0, 2 * np.pi, (self.layers, self.n_qubits)),
            "w": rng.normal(1.0, 0.3, (self.layers, self.n_qubits)),
            "b": rng.normal(0.0, 0.3, (self.layers, self.n_qubits)),
            "alpha": np.array(3.0),
        }

        def loss_b(params, Xb, yb):
            out = self._circuit(Xb, params)
            p = sigmoid(params["alpha"] * out)
            return bce_loss(p, yb)

        opt = Adam(self.params, lr=self.lr)
        best_loss, best_params, stall = np.inf, None, 0
        self.history = []
        for epoch in range(self.epochs):
            epoch_loss, nb = 0.0, 0
            for bidx in batches(len(y), self.batch_size, rng):
                Xb, yb = Xsq[bidx], y[bidx]
                g = grad(lambda pp: loss_b(pp, Xb, yb))(self.params)
                self.params = opt.step(self.params, g)
                epoch_loss += float(loss_b(self.params, Xb, yb))
                nb += 1
            epoch_loss /= max(1, nb)
            self.history.append(epoch_loss)
            if epoch_loss < best_loss - 1e-4:
                best_loss, best_params, stall = epoch_loss, dict(self.params), 0
            else:
                stall += 1
                if stall >= self.patience:
                    break
        if best_params is not None:
            self.params = best_params

    def _positive_proba(self, X: np.ndarray) -> np.ndarray:
        Xsq = quantum_input_transform(X)
        out = self._circuit(Xsq, self.params)
        p = np.asarray(sigmoid(self.params["alpha"] * out))
        return np.clip(p, 1e-6, 1 - 1e-6)

    def n_parameters(self) -> int:
        return count_params(self.params)

    def quantum_resources(self) -> Optional[Dict[str, Any]]:
        cir = BatchedCircuit(self.n_qubits)
        self._circuit(np.zeros((1, self.n_qubits)), self.params,
                      gate_counter=cir)
        depth = circuit_depth(cir.gate_counts, self.n_qubits,
                              entangling=self.entanglement)
        return resource_report(self.n_qubits, cir.gate_counts,
                               n_circuits=self.n_train_samples *
                               (len(self.history) or 1),
                               depth=depth, shots=self.shots)

    def metadata(self) -> Dict[str, Any]:
        m = super().metadata()
        m.update({
            "n_qubits": self.n_qubits, "layers": self.layers,
            "encoding": "data re-uploading", "backend": self.backend,
            "entanglement": self.entanglement,
            "epochs_run": len(self.history),
            "final_train_loss": float(self.history[-1]) if self.history else None,
            "readout": "mean Pauli-Z over all qubits",
        })
        return m
