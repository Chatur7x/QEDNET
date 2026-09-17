"""Quantum input sensitivity (D9)."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def quantum_input_sensitivity(
    circuit_fn,
    params: np.ndarray,
    x: np.ndarray,
    n_qubits: int,
    epsilon: float = 1e-3,
) -> dict:
    """
    Compute sensitivity of circuit output to each input feature.
    Returns gradient of output w.r.t. each input feature.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        circuit_fn(params, x)
        return qml.expval(qml.Z(0))

    # Compute gradients w.r.t. inputs
    x_tensor = pnp.array(x, requires_grad=True)
    output = circuit(x_tensor)
    grad_fn = qml.grad(lambda x: circuit(x))
    grads = grad_fn(x_tensor)

    return {
        "gradients": np.array(grads).tolist(),
        "mean_abs": float(np.mean(np.abs(grads))),
        "max_abs": float(np.max(np.abs(grads))),
        "feature_sensitivity": np.abs(grads).tolist(),
    }


def parameter_sensitivity(
    circuit_fn,
    params: np.ndarray,
    x: np.ndarray,
    n_qubits: int,
) -> dict:
    """Sensitivity to each trainable parameter."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(p, x):
        circuit_fn(p, x)
        return qml.expval(qml.Z(0))

    params_tensor = pnp.array(params, requires_grad=True)
    grad_fn = qml.grad(lambda p: circuit(p, x))
    grads = grad_fn(params_tensor)

    return {
        "gradients": np.array(grads).ravel().tolist(),
        "mean_abs": float(np.mean(np.abs(grads))),
        "max_abs": float(np.max(np.abs(grads))),
    }


def finite_difference_sensitivity(
    model,
    X: np.ndarray,
    feature_idx: int,
    epsilon: float = 1e-3,
) -> float:
    """Finite difference sensitivity for any model."""
    X_plus = X.copy()
    X_minus = X.copy()
    X_plus[:, feature_idx] += epsilon
    X_minus[:, feature_idx] -= epsilon

    pred_plus = model.predict_proba(X_plus)[:, 1]
    pred_minus = model.predict_proba(X_minus)[:, 1]

    return float(np.mean(np.abs(pred_plus - pred_minus) / (2 * epsilon)))