"""F5 — Variational Quantum Classifier (VQC).

Features -> quantum encoding -> variational circuit -> measurement ->
prediction. Trainable parameters, configurable layers/encoding/optimizer,
simulator execution (batched statevector or PennyLane reference backend).

Ansatz (real-unitary RY/CNOT so reverse-mode backprop is exact):
    RY(pi * squash(x_q)) per qubit
    [ RY(theta_l_q) per qubit ; CNOT ring ] * layers
    Pauli-Z readout on qubit 0 -> sigmoid(alpha * <Z>)
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
                                ladder_entanglers, resource_report)


class VQC(BaseModel):
    name = "vqc"
    family = "quantum"

    def __init__(self, random_state: int = 42, n_qubits: int = 8,
                 layers: int = 3, encoding: str = "angle",
                 backend: str = "numpy", entanglement: bool = True,
                 epochs: int = 60, batch_size: int = 32, lr: float = 0.08,
                 patience: int = 12, shots: Optional[int] = None, **kw):
        super().__init__(random_state)
        if encoding not in ("angle", "amplitude"):
            raise ValueError("VQC supports encoding 'angle' or 'amplitude'")
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

    # ------------------------------------------------------------- circuit
    def _circuit(self, Xsq: anp.ndarray, W: anp.ndarray, alpha: anp.ndarray,
                 gate_counter: Optional[BatchedCircuit] = None):
        """Forward pass: batched statevector evolution + Z readout."""
        B = Xsq.shape[0]
        cir = BatchedCircuit(self.n_qubits)
        state = cir.zero_state(B)
        # angle encoding: RY(pi * x_q)
        for q in range(self.n_qubits):
            state = cir.ry(state, q, anp.pi * Xsq[:, q % Xsq.shape[1]])
        # trainable layers
        ent = ladder_entanglers(self.n_qubits) if self.entanglement else []
        for l in range(self.layers):
            for q in range(self.n_qubits):
                state = cir.ry(state, q, W[l, q])
            if self.entanglement:
                for c, t in ent:
                    state = cir.cnot(state, c, t)
        out = cir.expval_z(state, 0)
        if gate_counter is not None:
            gate_counter.gate_counts.update(cir.gate_counts)
        return out, alpha

    # ------------------------------------------------------------------ fit
    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        Xsq = quantum_input_transform(X)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        W0 = rng.uniform(0, 2 * np.pi, (self.layers, self.n_qubits))
        self.params = {"W": W0, "alpha": np.array(2.0)}

        def make_logits(params):
            W = params["W"]
            alpha = params["alpha"]
            out, _ = self._circuit(Xsq, W, alpha)
            return alpha * out

        def loss_fn(params):
            logits = make_logits(params)
            p = sigmoid(logits)
            return bce_loss(p, y)

        opt = Adam(self.params, lr=self.lr)
        best_loss, best_params, stall = np.inf, None, 0
        self.history = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for bidx in batches(len(y), self.batch_size, rng):
                Xb, yb = Xsq[bidx], y[bidx]

                def loss_b(params, Xb=Xb, yb=yb):
                    out, _ = self._circuit(Xb, params["W"], params["alpha"])
                    p = sigmoid(params["alpha"] * out)
                    return bce_loss(p, yb)

                g = grad(loss_b)(self.params)
                self.params = opt.step(self.params, g)
                epoch_loss += float(loss_b(self.params))
                n_batches += 1
            epoch_loss /= max(1, n_batches)
            self.history.append(epoch_loss)
            if epoch_loss < best_loss - 1e-4:
                best_loss, best_params, stall = epoch_loss, dict(self.params), 0
            else:
                stall += 1
                if stall >= self.patience:
                    break
        if best_params is not None:
            self.params = best_params

    # ------------------------------------------------------------- predict
    def _positive_proba(self, X: np.ndarray) -> np.ndarray:
        Xsq = quantum_input_transform(X)
        out, alpha = self._circuit(Xsq, self.params["W"], self.params["alpha"])
        logits = alpha * out
        if self.shots is not None:
            # shot-noise on the measured expectation (deployment realism)
            p0 = (1.0 + out) / 2.0
            rng = np.random.default_rng(self.random_state + 1)
            sampled = rng.binomial(self.shots, np.clip(p0, 0, 1)) / self.shots
            out = 2.0 * sampled - 1.0
            logits = alpha * out
        p = np.asarray(sigmoid(logits))
        return np.clip(p, 1e-6, 1 - 1e-6)

    # ------------------------------------------------------------- metadata
    def n_parameters(self) -> int:
        return count_params(self.params)

    def quantum_resources(self) -> Optional[Dict[str, Any]]:
        cir = BatchedCircuit(self.n_qubits)
        self._circuit(np.zeros((1, self.n_qubits)), self.params["W"],
                      self.params["alpha"], gate_counter=cir)
        depth = circuit_depth(cir.gate_counts, self.n_qubits,
                              entangling=self.entanglement)
        return resource_report(self.n_qubits, cir.gate_counts,
                               n_circuits=self.n_train_samples *
                               (self.history and len(self.history) or 1),
                               depth=depth, shots=self.shots)

    def metadata(self) -> Dict[str, Any]:
        m = super().metadata()
        m.update({
            "n_qubits": self.n_qubits, "layers": self.layers,
            "encoding": self.encoding, "backend": self.backend,
            "entanglement": self.entanglement,
            "epochs_run": len(self.history),
            "final_train_loss": float(self.history[-1]) if self.history else None,
        })
        return m
