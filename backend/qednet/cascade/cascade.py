"""F12 — Classical-Quantum Second-Opinion Cascade.

User Input -> Classical Screening Model -> Confidence Gate
  * high confidence  -> classical result
  * low confidence   -> quantum model -> calibration -> uncertainty -> result

Default ambiguity band 0.35 <= p <= 0.65 (configurable; NOT assumed to be
universally optimal — the chosen threshold is always recorded).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from ..uncertainty.abstention import (AbstentionRule, ConformalClassifier,
                                      predictive_entropy)


def cascade_evaluate(p_screen: np.ndarray, p_quantum: np.ndarray,
                     y: np.ndarray, band_low: float = 0.35,
                     band_high: float = 0.65,
                     calibrator=None) -> Dict[str, Any]:
    """Evaluate the cascade on out-of-fold/screening probabilities.

    ``p_screen``: classical screening model probabilities (fast first stage)
    ``p_quantum``: quantum second-opinion probabilities for the SAME samples
    """
    y = np.asarray(y, int)
    ps = np.clip(np.asarray(p_screen, float), 0, 1)
    pq = np.clip(np.asarray(p_quantum, float), 0, 1)

    ambiguous = (ps >= band_low) & (ps <= band_high)
    # gate: high-confidence -> classical; ambiguous -> quantum second opinion
    p_final = np.where(ambiguous, pq, ps)
    if calibrator is not None:
        # calibrate only the quantum-routed subset (calibrator fit on train)
        if ambiguous.any():
            p_final[ambiguous] = calibrator.transform(pq[ambiguous])

    def _auc(yy, pp):
        mask = ~np.isnan(pp)
        if len(np.unique(yy[mask])) < 2:
            return None
        return float(roc_auc_score(yy[mask], pp[mask]))

    both = _auc(y, ps), _auc(y, pq), _auc(y, p_final)
    acc_screen = float(np.mean((ps >= 0.5).astype(int) == y))
    acc_quantum = float(np.mean((pq >= 0.5).astype(int) == y))
    acc_final = float(np.mean((p_final >= 0.5).astype(int) == y))

    routed = ambiguous
    routed_correct = float(np.mean(
        (p_final[routed] >= 0.5).astype(int) == y[routed])) if routed.any() else None
    direct_correct = float(np.mean(
        (p_final[~routed] >= 0.5).astype(int) == y[~routed])) if (~routed).any() else None

    return {
        "configuration": {
            "screening_model": "classical (configured)",
            "ambiguity_band": [float(band_low), float(band_high)],
            "note": ("threshold is configurable and recorded; the band is "
                     "not assumed universally optimal"),
        },
        "routing": {
            "n_total": int(len(y)),
            "n_routed_to_quantum": int(routed.sum()),
            "coverage_routed_fraction": float(routed.mean()),
            "n_classical_direct": int((~routed).sum()),
        },
        "performance": {
            "auroc_screening_alone": both[0],
            "auroc_quantum_alone": both[1],
            "auroc_cascade": both[2],
            "accuracy_screening_alone": acc_screen,
            "accuracy_quantum_alone": acc_quantum,
            "accuracy_cascade": acc_final,
            "accuracy_on_routed_subset": routed_correct,
            "accuracy_on_direct_subset": direct_correct,
        },
        "uncertainty": {
            "entropy_final": [round(float(v), 4)
                              for v in predictive_entropy(p_final)[:0]],
            "mean_entropy_screening": float(np.mean(predictive_entropy(ps))),
            "mean_entropy_quantum": float(np.mean(predictive_entropy(pq))),
        },
    }


def cascade_predict_single(x_row: np.ndarray, screening_model,
                           quantum_model, pipe, band_low: float = 0.35,
                           band_high: float = 0.65,
                           calibrator=None,
                           abstention: Optional[AbstentionRule] = None
                           ) -> Dict[str, Any]:
    """Single-sample cascade with full provenance for the Prediction view."""
    Z = pipe.transform(x_row.reshape(1, -1))
    p_classical = float(screening_model.predict_proba(Z)[:, 1][0])
    route = "classical"
    p_used, model_used = p_classical, getattr(screening_model, "name",
                                              "classical")
    if band_low <= p_classical <= band_high:
        route = "quantum"
        p_q = float(quantum_model.predict_proba(Z)[:, 1][0])
        p_used, model_used = p_q, getattr(quantum_model, "name", "quantum")
        if calibrator is not None:
            p_used = float(calibrator.transform(np.array([p_q]))[0])

    decision = {"abstain": False, "reasons": {}}
    if abstention is not None:
        d = abstention.decide(np.array([p_used]))
        decision = {
            "abstain": bool(d["abstain"][0]),
            "reasons": {k: bool(v[0]) for k, v in d["reasons"].items()},
        }

    return {
        "stage": "cascade",
        "route": route,
        "classical_screening_probability": round(p_classical, 6),
        "model_used": model_used,
        "probability_used": round(p_used, 6),
        "prediction": None if decision["abstain"]
        else int(p_used >= 0.5),
        "confidence": round(max(p_used, 1 - p_used), 6),
        "uncertainty_entropy": round(float(predictive_entropy(
            np.array([p_used]))[0]), 6),
        "abstention": decision,
        "status": "abstention triggered — prediction uncertain" if
        decision["abstain"] else "prediction available",
    }
