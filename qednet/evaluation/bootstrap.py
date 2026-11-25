"""Bootstrap utilities (D8)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_statistic(
    data: np.ndarray,
    statistic_fn: Callable,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Generic bootstrap for any statistic."""
    rng = np.random.RandomState(random_state)
    n = len(data)
    point = statistic_fn(data)
    bootstraps = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            bootstraps.append(statistic_fn(data[idx]))
        except Exception:
            continue

    if len(bootstraps) < 10:
        return {
            "point": point,
            "ci_lower": point,
            "ci_upper": point,
            "n_bootstraps": len(bootstraps),
        }

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(bootstraps, 100 * alpha))
    upper = float(np.percentile(bootstraps, 100 * (1 - alpha)))
    return {
        "point": float(point),
        "ci_lower": lower,
        "ci_upper": upper,
        "std": float(np.std(bootstraps)),
        "n_bootstraps": len(bootstraps),
    }


def bootstrap_ci_diff(
    data1: np.ndarray,
    data2: np.ndarray,
    statistic_fn: Callable,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Bootstrap CI for difference of statistics."""
    rng = np.random.RandomState(random_state)
    n = len(data1)
    point = statistic_fn(data1) - statistic_fn(data2)
    bootstraps = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            bootstraps.append(statistic_fn(data1[idx]) - statistic_fn(data2[idx]))
        except Exception:
            continue

    if len(bootstraps) < 10:
        return {"point": point, "ci_lower": point, "ci_upper": point}

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(bootstraps, 100 * alpha))
    upper = float(np.percentile(bootstraps, 100 * (1 - alpha)))
    return {
        "point": float(point),
        "ci_lower": lower,
        "ci_upper": upper,
        "std": float(np.std(bootstraps)),
    }
