"""Quantum Advantage Certificate generation from actual experiment data (F9).

Categories (deterministic, documented):
- "Parameter-Efficiency Advantage": quantum beats the parameter-matched MLP
  with non-overlapping 95% CIs.
- "Small-Data Advantage": n_train <= 200 and quantum beats best classical
  with non-overlapping 95% CIs.
- "Inconclusive / Underpowered": CIs overlap (cannot tell).
- "No Demonstrated Advantage": otherwise.
Never hard-coded: derived from the metrics passed in.
"""

from __future__ import annotations

import hashlib
import json


def _ci_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[0] > b[1] or b[0] > a[1])


def generate_certificate(
    experiment_id: str,
    dataset: str,
    model: str,
    config_hash: str,
    seeds: list[int],
    quantum: dict,
    classical_best: dict,
    param_matched: dict,
    ablation_delta: float | None = None,
    dequantization: dict | None = None,
    resources: dict | None = None,
    n_train: int = 0,
) -> dict:
    """Build a certificate dict. Metric dicts need {mean, ci_lower, ci_upper}."""
    q_ci = (quantum["ci_lower"], quantum["ci_upper"])
    c_ci = (classical_best["ci_lower"], classical_best["ci_upper"])
    p_ci = (param_matched["ci_lower"], param_matched["ci_upper"])
    overlap_qc = _ci_overlap(q_ci, c_ci)

    if quantum["mean"] > param_matched["mean"] and not _ci_overlap(q_ci, p_ci):
        verdict = "Parameter-Efficiency Advantage"
    elif n_train and n_train <= 200 and quantum["mean"] > classical_best["mean"] and not overlap_qc:
        verdict = "Small-Data Advantage"
    elif overlap_qc:
        verdict = "Inconclusive / Underpowered"
    else:
        verdict = "No Demonstrated Advantage"

    cert = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model": model,
        "configuration_hash": config_hash,
        "seeds": seeds,
        "quantum": quantum,
        "classical_best": classical_best,
        "param_matched": param_matched,
        "ablation_delta": ablation_delta,
        "dequantization": dequantization or {},
        "resources": resources or {},
        "n_train": n_train,
        "evidence_classification": verdict,
        "safety_note": "Research use only. Not a medical device; not for diagnosis.",
    }
    cert["certificate_hash"] = hashlib.sha256(
        json.dumps(cert, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return cert


def certificate_to_markdown(cert: dict) -> str:
    lines = [
        f"# Quantum Advantage Certificate — {cert['experiment_id']}",
        "",
        f"- Dataset: {cert['dataset']}",
        f"- Model: {cert['model']}",
        f"- Config hash: {cert['configuration_hash']}",
        f"- Seeds: {cert['seeds']}",
        f"- Quantum AUROC: {cert['quantum']['mean']:.4f} "
        f"[{cert['quantum']['ci_lower']:.4f}, {cert['quantum']['ci_upper']:.4f}]",
        f"- Best classical AUROC: {cert['classical_best']['mean']:.4f} "
        f"[{cert['classical_best']['ci_lower']:.4f}, {cert['classical_best']['ci_upper']:.4f}]",
        f"- Ablation delta: {cert['ablation_delta']}",
        f"- Verdict: **{cert['evidence_classification']}**",
        "",
        cert["safety_note"],
    ]
    return "\n".join(lines)
