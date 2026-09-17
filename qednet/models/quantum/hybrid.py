"""Hybrid QNN quantum layer (D6) - PennyLane TorchLayer compatible.

Architecture: classical features (8-d) -> VQC quantum layer -> dense head.
This module provides only the quantum layer (TorchLayer), not the full data pipeline.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

# PennyLane >=0.35 uses qnn.TorchLayer
try:
    from pennylane.qnn import TorchLayer
except Exception:
    try:
        from pennylane.nn import TorchLayer
    except Exception:
        TorchLayer = None


def _make_vqc_torqlayer(n_qubits: int, n_layers: int, encoding: str = "angle",
                        device: str = "default.qubit", diff_method: str = "backprop"):
    """Create a PennyLane TorchLayer wrapping a VQC-like circuit."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available; install torch to use HybridQNNLayer")
    if TorchLayer is None:
        raise RuntimeError("PennyLane TorchLayer not available (need pennylane>=0.35)")

    if diff_method.lower() in ("backprop", "back-prop"):
        diff = "adjoint" if "lightning" in device else "backprop"
    elif diff_method.lower() in ("parameter-shift", "ps"):
        diff = "parameter-shift"
    else:
        diff = "finite-diff"

    kwargs = {"wires": n_qubits}
    try:
        dev = qml.device(device, **kwargs)
    except Exception:
        dev = qml.device("default.qubit", **kwargs)
        device = "default.qubit"

    @qml.qnode(dev, diff_method=diff, interface="torch")
    def circuit(inputs, weights):
        # inputs: batch x n_qubits
        # weights: (n_layers, n_qubits, 3)
        for i in range(n_qubits):
            qml.RX(inputs[:, i], wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1],
                        weights[layer, i, 2], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        # return as 1D array to match TorchLayer expectations
        return qml.expval(qml.Z(0))

    # weight shapes for TorchLayer
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    torch_layer = TorchLayer(circuit, weight_shapes)

    # wrapper to ensure output is (batch, 1)
    class _WrappedLayer(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            out = self.layer(x)
            # TorchLayer returns (batch,) -> reshape to (batch, 1)
            if out.dim() == 1:
                out = out.unsqueeze(-1)
            return out

    return _WrappedLayer(torch_layer), dev, circuit


class HybridQNNLayer(nn.Module if _TORCH_AVAILABLE else object):
    """PyTorch module: quantum layer for hybrid QNN.

    Forward: (batch, n_qubits) -> (batch, 1) expectation value.
    Compatible with nn.Sequential: Dense -> HybridQNNLayer -> Dense -> Sigmoid.
    """
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        encoding: str = "angle",
        device: str = "default.qubit",
        diff_method: str = "backprop",
        seed: int = 0,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for HybridQNNLayer")
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.encoding = encoding.lower()
        self.device_name = device
        self.diff_method = diff_method
        self.seed = int(seed)

        self._qnode, self._dev, self._circuit = _make_vqc_torqlayer(
            n_qubits, n_layers, encoding, device, diff_method
        )

        # expose parameter count
        self._n_params = n_layers * n_qubits * 3

    @property
    def n_params(self) -> int:
        return self._n_params

    def forward(self, x):
        # x: (batch, n_qubits) -> (batch, 1)
        return self._qnode(x)

    def metadata(self) -> dict:
        two_q = (self.n_qubits if self.n_qubits > 1 else 0) * self.n_layers
        return {
            "model": "hybrid_qnn_layer",
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "encoding": self.encoding,
            "n_params": self.n_params,
            "two_qubit_gates": two_q,
            "device": self.device_name,
            "diff_method": self.diff_method,
        }


def make_hybrid_model(input_dim: int = 64, n_qubits: int = 8, n_layers: int = 2,
                      hidden_dim: int = 32, encoding: str = "angle",
                      device: str = "default.qubit", diff_method: str = "backprop",
                      seed: int = 0):
    """Build a complete hybrid model: Linear -> HybridQNNLayer -> Linear -> Sigmoid.

    Member C owns the training pipeline; this is a convenience factory.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    return nn.Sequential(
        nn.Linear(input_dim, n_qubits),
        nn.Tanh(),  # bound inputs to [-1, 1] for angle encoding
        HybridQNNLayer(n_qubits, n_layers, encoding, device, diff_method, seed),
        nn.Linear(1, 1),
        nn.Sigmoid(),
    )


def gradient_propagation_test(n_qubits: int = 4, n_layers: int = 1,
                              device: str = "default.qubit") -> bool:
    """Test that gradients flow through the quantum layer."""
    if not _TORCH_AVAILABLE:
        return False
    import torch

    layer = HybridQNNLayer(n_qubits, n_layers, device=device)
    x = torch.randn(2, n_qubits, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    return x.grad is not None and x.grad.abs().sum() > 0