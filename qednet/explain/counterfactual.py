"""Counterfactual explanations (D9)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def find_counterfactual(
    model,
    x: np.ndarray,
    target_class: int = 1,
    max_iter: int = 100,
    step_size: float = 0.1,
    lambda_reg: float = 1.0,
) -> dict:
    """
    Find minimal perturbation to change prediction.
    Uses gradient-based optimization on model's decision boundary.
    """
    x = np.array(x, dtype=float).ravel()
    x0 = x.copy()

    def objective(x_new):
        # Distance from original
        dist = np.sum((x_new - x0) ** 2)
        # Prediction loss
        prob = model.predict_proba(x_new.reshape(1, -1))[0, target_class]
        # Want prob -> 1 for target class
        loss = -np.log(prob + 1e-10)
        return dist + lambda_reg * loss

    # Gradient-free optimization
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": max_iter, "xatol": 1e-6})

    x_cf = result.x
    prob_cf = model.predict_proba(x_cf.reshape(1, -1))[0, target_class]
    pred_cf = (prob_cf >= 0.5).astype(int)

    return {
        "original": x0.tolist(),
        "counterfactual": x_cf.tolist(),
        "target_class": target_class,
        "cf_probability": float(prob_cf),
        "cf_prediction": int(pred_cf),
        "distance": float(np.sum((x_cf - x0) ** 2)),
        "feature_changes": (x_cf - x0).tolist(),
        "success": pred_cf == target_class,
        "optimization_success": result.success,
    }


def generate_counterfactuals(
    model,
    X: np.ndarray,
    n_samples: int = 10,
    target_class: int = 1,
) -> list[dict]:
    """Generate counterfactuals for multiple samples."""
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    results = []
    for i in idx:
        cf = find_counterfactual(model, X[i], target_class)
        cf["sample_idx"] = int(i)
        results.append(cf)
    return results