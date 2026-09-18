"""F5 — Hybrid QNN.

Classical feature extractor -> feature bottleneck -> quantum layer ->
classical prediction head -> prediction. The classical extractor (a
trainable linear+tanh layer) compresses features before the quantum
component; the classical head maps the quantum readout to a probability.
All parameters train jointly end-to-end (Adam + BCE) through the batched
statevector simulator.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import autograd.numpy as anp
import numpy as np
from autograd import grad

from ...models.base import BaseModel
from ...models.training import (Adam, bce_loss, batches, count_params,
                                sigmoid)
from ...sim.statevector import (BatchedCircuit, circuit_depth,
                                ladder_entanglers, resource_report)


class HybridQNN(BaseModel):
    name = "hybrid_qnn"
    family = "quantum"

    def __init__(self, random_state: int = 42, n_qubits: int = 8,
                 layers: int = 3, encoding: str = "hybrid_dense",
                 backend: str = "numpy", entanglement: bool = True,
                 epochs: int = 60, batch_size: int = 32, lr: float = 0.05,
                 patience: int = 12, shots: Optional[int] = None, **kw):
        super().__init__(random_state)
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
        self.input_dim: int = 0

    def _forward(self, X: anp.ndarray, P: Dict[str, anp.ndarray],
                 gate_counter: Optional[BatchedCircuit] = None):
        """Classical extractor -> quantum layer -> classical head."""
        # classical feature extractor (bottleneck to n_qubits)
        H = anp.tanh(X @ P["Wc"] + P["bc"])          # (B, n_qubits)
        Xsq = 0.5 * (H + 1.0)                         # squash to [0, 1]
        # quantum layer: RY-CNOT ansatz
        B = Xsq.shape[0]
        cir = BatchedCircuit(self.n_qubits)
        state = cir.zero_state(B)
        for q in range(self.n_qubits):
            state = cir.ry(state, q, anp.pi * Xsq[:, q])
        ent = ladder_entanglers(self.n_qubits) if self.entanglement else []
        for l in range(self.layers):
            for q in range(self.n_qubits):
                state = cir.ry(state, q, P["W"][l, q])
            if self.entanglement:
                for c, t in ent:
                    state = cir.cnot(state, c, t)
        z = cir.mean_z(state)                         # (B,)
        if gate_counter is not None:
            gate_counter.gate_counts.update(cir.gate_counts)
        # classical prediction head
        logits = P["out_w"] * z + P["out_b"]
        return logits

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.input_dim = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        scale = 1.0 / np.sqrt(self.input_dim)
        self.params = {
            "Wc": rng.normal(0, scale, (self.input_dim, self.n_qubits)),
            "bc": np.zeros(self.n_qubits),
            "W": rng.uniform(0, 2 * np.pi, (self.layers, self.n_qubits)),
            "out_w": np.array(2.0),
            "out_b": np.array(0.0),
        }

        def loss_b(params, Xb, yb):
            logits = self._forward(Xb, params)
            return bce_loss(sigmoid(logits), yb)

        opt = Adam(self.params, lr=self.lr)
        best_loss, best_params, stall = np.inf, None, 0
        self.history = []
        for epoch in range(self.epochs):
            epoch_loss, nb = 0.0, 0
            for bidx in batches(len(y), self.batch_size, rng):
                Xb, yb = X[bidx], y[bidx]
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
        logits = self._forward(np.asarray(X, dtype=np.float64), self.params)
        p = np.asarray(sigmoid(logits))
        return np.clip(p, 1e-6, 1 - 1e-6)

    def n_parameters(self) -> int:
        return count_params(self.params)

    def quantum_resources(self) -> Optional[Dict[str, Any]]:
        cir = BatchedCircuit(self.n_qubits)
        X0 = np.zeros((1, max(1, self.input_dim)))
        self._forward(X0, self.params, gate_counter=cir)
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
            "encoding": "classical dense bottleneck -> angle",
            "backend": self.backend, "entanglement": self.entanglement,
            "classical_extractor": f"dense {self.input_dim}->{self.n_qubits} tanh",
            "epochs_run": len(self.history),
            "final_train_loss": float(self.history[-1]) if self.history else None,
        })
        return m
