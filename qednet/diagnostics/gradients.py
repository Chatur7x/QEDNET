"""Gradient diagnostics (D7).

- gradient_variance: compute variance of gradients over parameters
- barren_plateau_diagnostic: structured result with criterion documentation

Does NOT declare "barren plateau" from a single number.
Documents the diagnostic criterion used.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def gradient_variance(circuit_fn, params, *args, **kwargs) -> dict:
    """Compute gradient variance for a parameterized circuit.

    Args:
        circuit_fn: QNode or function returning expectation value
        params: parameter array (will be converted to pnp.array with requires_grad)
        *args, **kwargs: additional inputs to circuit_fn

    Returns:
        dict with:
        - gradients: array of gradient values
        - variance: float
        - mean_abs: float
        - max_abs: float
        - n_params: int
    """
    params = pnp.array(params, requires_grad=True)

    # Wrap circuit to take params as first arg
    def _loss(p):
        return circuit_fn(*args, p, **kwargs)

    # Compute gradient
    grad_fn = qml.grad(_loss)
    grads = grad_fn(params)

    # Handle different return types
    grads = pnp.asarray(grads).ravel()

    return {
        "gradients": grads,
        "variance": float(pnp.var(grads)),
        "mean_abs": float(pnp.mean(pnp.abs(grads))),
        "max_abs": float(pnp.max(pnp.abs(grads))),
        "n_params": len(grads),
    }


def barren_plateau_diagnostic(circuit_fn, params, *args,
                               n_samples: int = 10,
                               threshold: float = 1e-3,
                               **kwargs) -> dict:
    """Barren plateau diagnostic (McClean et al. 2018 criterion).

    Criterion: Variance of gradients decays exponentially with n_qubits.
    For random circuits: Var[∂C/∂θ] ~ O(1/2^n) for deep circuits.
    We estimate variance at multiple random parameter initializations.

    Does NOT declare barren plateau from a single number.
    Returns structured data for human/MLflow evaluation.

    Args:
        circuit_fn: function(params, *args) -> scalar cost
        params: initial parameter array (shape used for sampling)
        n_samples: number of random initializations to sample
        threshold: variance below this *may* indicate barren plateau
                   (not a definitive threshold; problem-dependent)
        **kwargs: passed to circuit_fn

    Returns:
        dict with:
        - variances: list of variances per sample
        - mean_variance: float
        - std_variance: float
        - threshold: float
        - n_samples: int
        - criterion: str (description of diagnostic used)
        - note: str (interpretation guidance)
    """
    params = pnp.asarray(params)
    shape = params.shape
    variances = []

    for _ in range(n_samples):
        # Sample random parameters from uniform distribution
        sample = pnp.array(np.random.uniform(-np.pi, np.pi, shape), requires_grad=True)

        def _loss(p):
            return circuit_fn(p, *args, **kwargs)

        grad_fn = qml.grad(_loss)
        grads = grad_fn(sample)
        grads = pnp.asarray(grads).ravel()
        variances.append(float(pnp.var(grads)))

    mean_var = float(np.mean(variances))
    std_var = float(np.std(variances))

    return {
        "variances": variances,
        "mean_variance": mean_var,
        "std_variance": std_var,
        "threshold": threshold,
        "n_samples": n_samples,
        "criterion": (
            "Gradient variance decays exponentially with qubit count for "
            "random deep circuits (McClean et al. 2018). "
            f"Threshold {threshold} is a heuristic, not a definitive test."
        ),
        "note": (
            "Low variance suggests potential barren plateau but depends on "
            "circuit architecture, encoding, and problem. "
            "Compare against shallow/classical baselines."
        ),
    }