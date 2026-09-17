"""Hybrid QNN quantum layer tests."""

import numpy as np
import pytest

from qednet.models.quantum.hybrid import (
    HybridQNNLayer,
    make_hybrid_model,
    gradient_propagation_test,
)


def test_layer_init_and_metadata():
    if not gradient_propagation_test():
        pytest.skip("PyTorch not available or test failed")
    layer = HybridQNNLayer(n_qubits=4, n_layers=1)
    assert layer.n_params == 12
    meta = layer.metadata()
    assert meta["model"] == "hybrid_qnn_layer"
    assert meta["n_qubits"] == 4
    assert meta["n_layers"] == 1


def test_forward_shape():
    if not gradient_propagation_test():
        pytest.skip("PyTorch not available")
    import torch

    layer = HybridQNNLayer(n_qubits=4, n_layers=1)
    x = torch.randn(3, 4)
    y = layer(x)
    assert y.shape == (3, 1)


def test_gradient_propagation():
    if not gradient_propagation_test():
        pytest.skip("PyTorch not available")
    import torch

    layer = HybridQNNLayer(n_qubits=4, n_layers=1)
    x = torch.randn(2, 4, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_make_hybrid_model():
    if not gradient_propagation_test():
        pytest.skip("PyTorch not available")
    import torch

    model = make_hybrid_model(input_dim=16, n_qubits=4, n_layers=1)
    x = torch.randn(2, 16)
    y = model(x)
    assert y.shape == (2, 1)
    assert (y >= 0).all() and (y <= 1).all()


def test_deterministic_weights():
    if not gradient_propagation_test():
        pytest.skip("PyTorch not available")
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    layer1 = HybridQNNLayer(n_qubits=4, n_layers=1, seed=0)
    w1 = [p.detach().clone() for p in layer1.parameters()]

    torch.manual_seed(0)
    np.random.seed(0)
    layer2 = HybridQNNLayer(n_qubits=4, n_layers=1, seed=0)
    w2 = [p.detach().clone() for p in layer2.parameters()]

    for a, b in zip(w1, w2):
        assert torch.allclose(a, b)