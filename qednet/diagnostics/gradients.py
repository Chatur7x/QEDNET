"""Gradient diagnostics (D7)."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def gradient_variance(circuit_fn, params, *args, **kwargs) -> dict:
    params = pnp.array(params, requires_grad=True)

    def _loss(p):
        return circuit_fn(*args, p, **kwargs)

    grad_fn = qml.grad(_loss)
    grads = pnp.asarray(grad_fn(params)).ravel()

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
    params = pnp.asarray(params)
    shape = params.shape
    variances = []

    for _ in range(n_samples):
        sample = pnp.array(np.random.uniform(-np.pi, np.pi, shape), requires_grad=True)

        def _loss(p):
            return circuit_fn(p, *args, **kwargs)

        grad_fn = qml.grad(_loss)
        grads = pnp.asarray(grad_fn(sample)).ravel()
        variances.append(float(pnp.var(grads)))

    mean_var = float(np.mean(variances))
    std_var = float(np.std(variances))

    return {
        "variances": variances,
        "mean_variance": mean_var,
        "std_variance": std_var,
        "threshold": threshold,
        "n_samples": n_samples,
        "criterion": "Gradient variance decays exponentially with qubit count for random deep circuits (McClean et al. 2018). Threshold is a heuristic.",
        "note": "Low variance suggests potential barren plateau; compare against shallow/classical baselines.",
    }