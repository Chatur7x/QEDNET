"""F10 — Bottleneck Honesty Meter.

How much of the predictive performance comes from classical feature
compression versus the quantum layer? Train a simple classical model on the
EXACT compressed representation the quantum model receives (same pipeline,
same squash) and compare against the quantum model on the same folds.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..models.training import quantum_input_transform


def bottleneck_honesty_meter(pipe, quantum_model, X_tr, y_tr, X_te, y_te,
                             seed: int = 42) -> Dict[str, Any]:
    """Compare simple-classical-on-compressed vs quantum-on-compressed."""
    Z_tr = pipe.transform(X_tr)
    Z_te = pipe.transform(X_te)

    # the exact representation the quantum layer consumes (post-squash)
    Q_tr = quantum_input_transform(Z_tr)
    Q_te = quantum_input_transform(Z_te)

    simple = LogisticRegression(max_iter=2000, random_state=seed)
    simple.fit(Q_tr, y_tr)
    p_simple = simple.predict_proba(Q_te)[:, 1]

    p_quantum = quantum_model.predict_proba(Z_te)[:, 1]

    def _auc(y, p):
        return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None

    auc_simple, auc_quantum = _auc(y_te, p_simple), _auc(y_te, p_quantum)
    gap = (auc_quantum - auc_simple) if (auc_quantum is not None
                                         and auc_simple is not None) else None
    if gap is None:
        reading = "inconclusive"
    elif gap > 0.01:
        reading = ("quantum layer adds measurable value beyond the "
                   "classical compressed representation")
    elif gap < -0.01:
        reading = ("simple classical model on the same compressed "
                   "representation outperforms the quantum layer")
    else:
        reading = ("most predictive performance comes from the classical "
                   "compression; quantum layer adds no measurable value")

    return {
        "compressed_space_baseline": {
            "model": "logistic regression on squashed compressed features",
            "auroc": auc_simple,
            "n_dims": int(Z_tr.shape[1]),
        },
        "quantum_score": {"model": getattr(quantum_model, "name", "quantum"),
                          "auroc": auc_quantum},
        "performance_gap_quantum_minus_baseline": None if gap is None
        else round(gap, 6),
        "reading": reading,
        "purpose": (
            "measures how much predictive performance comes from classical "
            "feature compression versus the quantum layer"),
    }
