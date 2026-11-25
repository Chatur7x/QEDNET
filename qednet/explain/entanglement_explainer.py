"""Entanglement explainer (D9)."""

from __future__ import annotations

import numpy as np

from qednet.diagnostics.entanglement import measure_entanglement


def entanglement_explainer(
    circuit_fn,
    params: np.ndarray,
    X: np.ndarray,
    n_qubits: int,
) -> dict:
    """
    Analyze entanglement in the quantum model for explainability.
    Returns per-sample entanglement measures.
    """
    results = []
    for x in X:
        ent = measure_entanglement(circuit_fn, params, x, n_qubits=n_qubits)
        results.append(
            {
                "entanglement_entropies": ent["entanglement_entropies"],
                "meyer_wallach": ent["meyer_wallach"],
            }
        )

    avg_mw = np.mean([r["meyer_wallach"] for r in results])
    avg_entropy = np.mean([np.mean(list(r["entanglement_entropies"].values())) for r in results])

    return {
        "per_sample": results,
        "avg_meyer_wallach": float(avg_mw),
        "avg_entanglement_entropy": float(avg_entropy),
        "interpretation": "High entanglement suggests quantum correlations are used"
        if avg_mw > 0.5
        else "Low entanglement",
    }


def circuit_entanglement_profile(
    circuit_fn,
    params: np.ndarray,
    n_qubits: int,
    n_samples: int = 10,
) -> dict:
    """Profile entanglement across random inputs."""
    rng = np.random.RandomState(42)
    profiles = []
    for _ in range(n_samples):
        x = rng.uniform(-np.pi, np.pi, n_qubits)
        ent = measure_entanglement(circuit_fn, params, x, n_qubits=n_qubits)
        profiles.append(ent["meyer_wallach"])

    return {
        "meyer_wallach_samples": profiles,
        "mean": float(np.mean(profiles)),
        "std": float(np.std(profiles)),
    }
