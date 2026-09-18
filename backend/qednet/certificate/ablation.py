"""Entanglement ablation with parameter-matched control (Section 19).

Compares the quantum model WITH entanglement against a parameter-matched
NON-entangled model (same ansatz, same parameter count, CNOTs removed and
replaced by additional local rotations so capacity is preserved). Removing
entanglement must NOT reduce parameter capacity, or the ablation is invalid.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from ..models.registry import build_model


def entanglement_ablation(model_key: str, X_tr, y_tr, X_te, y_te,
                          model_kwargs: Dict[str, Any],
                          seed: int = 42) -> Dict[str, Any]:
    """Train entangled vs parameter-matched non-entangled variants.

    Parameter matching is exact by construction: CNOT entanglers carry ZERO
    trainable parameters, so the non-entangled variant (``entanglement=False``
    removes only the CNOT gates) has an identical trainable-parameter count.
    Removing entangling gates therefore does not reduce parameter capacity —
    exactly the control the ablation requires.
    """
    if model_key not in ("vqc", "reupload", "hybrid_qnn"):
        return {"available": False,
                "reason": "ablation applies to variational circuit models"}

    kw_ent = dict(model_kwargs)
    kw_ent.update({"entanglement": True, "random_state": seed})
    m_ent = build_model(model_key, **kw_ent)
    m_ent.fit(X_tr, y_tr)
    p_ent = m_ent.predict_proba(X_te)[:, 1]

    kw_noent = dict(model_kwargs)
    kw_noent.update({"entanglement": False, "random_state": seed,
                     "layers": model_kwargs.get("layers", 3)})
    m_no = build_model(model_key, **kw_noent)
    m_no.fit(X_tr, y_tr)
    p_no = m_no.predict_proba(X_te)[:, 1]

    def _auc(y, p):
        return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None

    params_ent = m_ent.n_parameters()
    params_no = m_no.n_parameters()
    auc_e, auc_n = _auc(y_te, p_ent), _auc(y_te, p_no)
    delta = (auc_e - auc_n) if (auc_e is not None and auc_n is not None) else None

    if delta is None:
        contribution = "inconclusive"
    elif delta > 0.01:
        contribution = "positive contribution"
    elif delta < -0.01:
        contribution = "negative contribution"
    else:
        contribution = "no measurable contribution"

    return {
        "available": True,
        "model": model_key,
        "entangled": {
            "auroc": auc_e, "n_parameters": params_ent,
            "circuit_depth": (m_ent.quantum_resources() or {}).get("circuit_depth"),
            "entangling_gates": (m_ent.quantum_resources() or {})
            .get("gate_counts", {}).get("cnot", 0),
        },
        "non_entangled_parameter_matched": {
            "auroc": auc_n, "n_parameters": params_no,
            "circuit_depth": (m_no.quantum_resources() or {}).get("circuit_depth"),
            "entangling_gates": (m_no.quantum_resources() or {})
            .get("gate_counts", {}).get("cnot", 0),
        },
        "auroc_delta_entangled_minus_nonentangled": None if delta is None
        else round(delta, 6),
        "capacity_preserved": bool(params_no >= params_ent),
        "entanglement_contribution": contribution,
        "interpretation": (
            "Same-parameter-count comparison; differences within noise are "
            "reported as no measurable contribution. The result is whatever "
            "the experiment actually shows."),
    }
