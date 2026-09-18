"""F9 — Quantum Advantage Certificate.

Generated ONLY from actual experiment data. Evaluates:
  1. classical baseline performance        5. classical dequantization
  2. quantum model performance             6. statistical evidence
  3. parameter-matched model               7. computational cost
  4. entanglement ablation                 8. reproducibility info

Classification rules (documented, deterministic, never hard-coded):
  * No Demonstrated Advantage — best quantum not significantly better than
    best classical (DeLong p >= 0.05 and/or corrected-CV p >= 0.05).
  * Parameter-Efficiency Advantage — quantum matches best classical
    performance (within CI) using far fewer parameters than the matched MLP.
  * Small-Data Advantage — advantage appears only on the smallest dataset.
  * Inconclusive / Underpowered — evidence conflicts or CIs are too wide.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _mean(models: Dict[str, Any], key: str, metric: str = "auroc") -> Optional[float]:
    m = models.get(key)
    if not m:
        return None
    v = m["summary"][metric]["mean"]
    return None if v is None else float(v)


def _ci(models, key, metric="auroc"):
    m = models.get(key)
    if not m or not m.get("bootstrap_auroc"):
        return None
    bs = m["bootstrap_auroc"]
    return bs.get("ci_low"), bs.get("ci_high")


def classify_evidence(state: Dict[str, Any]) -> Dict[str, str]:
    """Deterministic evidence classification from measured quantities."""
    q_best = state.get("quantum_best_auroc")
    c_best = state.get("classical_best_auroc")
    delong_p = state.get("delong_p")
    cv_p = state.get("corrected_cv_p")
    param_ratio = state.get("quantum_vs_matched_mlp_param_ratio")
    n_samples = state.get("n_samples")

    if q_best is None or c_best is None:
        return {"classification": "Inconclusive / Underpowered",
                "rationale": "missing performance evidence"}

    sig_delong = (delong_p is not None and delong_p < 0.05)
    sig_cv = (cv_p is not None and cv_p < 0.05)
    q_wins = q_best > c_best

    if (sig_delong or sig_cv) and q_wins:
        if param_ratio is not None and param_ratio < 0.2:
            return {"classification": "Parameter-Efficiency Advantage",
                    "rationale": (
                        f"quantum significantly better (p<0.05) with "
                        f"{1/param_ratio:.1f}x fewer parameters than the "
                        "matched MLP")}
        if n_samples is not None and n_samples < 250:
            return {"classification": "Small-Data Advantage",
                    "rationale": "significant advantage on a small-n dataset"}
        return {"classification": "No Demonstrated Advantage",
                "rationale": (
                    "statistical significance present but no certified "
                    "advantage dimension; treat as advantage-candidate "
                    "pending replication")}

    if not q_wins:
        return {"classification": "No Demonstrated Advantage",
                "rationale": (
                    f"best quantum AUROC {q_best:.4f} <= best classical "
                    f"{c_best:.4f}; classical models remain competitive or "
                    "superior on this dataset")}

    return {"classification": "Inconclusive / Underpowered",
            "rationale": (
                f"quantum point estimate higher ({q_best:.4f} vs "
                f"{c_best:.4f}) but NOT statistically significant "
                f"(DeLong p={delong_p}, corrected-CV p={cv_p}); more "
                "seeds/samples needed before claiming advantage")}


def build_certificate(result: Dict[str, Any], ablation: Optional[Dict],
                      dequant: Optional[Dict], honesty: Optional[Dict],
                      config: Dict[str, Any],
                      reproducibility: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the certificate from actual experiment data."""
    models = result["models"]
    quantum_keys = [k for k, v in models.items() if v.get("family") == "quantum"]
    classical_keys = [k for k, v in models.items() if v.get("family") == "classical"]

    q_best_key = max(
        (k for k in quantum_keys), key=lambda k: _mean(models, k) or -1,
        default=None)
    c_best_key = max(
        (k for k in classical_keys), key=lambda k: _mean(models, k) or -1,
        default=None)

    q_best = _mean(models, q_best_key) if q_best_key else None
    c_best = _mean(models, c_best_key) if c_best_key else None

    stats = result.get("statistics", {}).get(q_best_key or "", {})
    delong = stats.get("delong_vs_best_classical") or {}
    cv_t = stats.get("corrected_cv_vs_best_classical") or {}

    q_params = models.get(q_best_key, {}).get("param_count")
    mlp_params = models.get("matched_mlp", {}).get("param_count")
    param_ratio = None
    if q_params and mlp_params:
        param_ratio = q_params / mlp_params

    state = {
        "quantum_best_auroc": q_best,
        "classical_best_auroc": c_best,
        "delong_p": delong.get("p_value"),
        "corrected_cv_p": cv_t.get("p_value"),
        "quantum_vs_matched_mlp_param_ratio": param_ratio,
        "n_samples": result.get("n_samples"),
    }
    verdict = classify_evidence(state)

    cert = {
        "certificate_title": "QED-Net 2.0 Quantum Advantage Certificate",
        "research_status": (
            "Research and educational platform — not a medical device, not "
            "clinically validated, not for diagnosis."),
        "experiment_id": config.get("experiment_id"),
        "dataset": result.get("dataset"),
        "model": q_best_key,
        "configuration_hash": config.get("config_hash"),
        "random_seeds": config.get("seeds"),
        "metrics": {
            "quantum_best": {
                "model": q_best_key, "auroc": q_best,
                "ci95": _ci(models, q_best_key) if q_best_key else None},
            "classical_best": {
                "model": c_best_key, "auroc": c_best,
                "ci95": _ci(models, c_best_key) if c_best_key else None},
            "all_models": {
                k: models[k]["summary"]["auroc"]["mean"] for k in models},
        },
        "confidence_intervals": {
            k: _ci(models, k) for k in models},
        "parameter_count": {
            "quantum": q_params,
            "matched_mlp": mlp_params,
            "ratio_quantum_over_mlp": round(param_ratio, 4)
            if param_ratio else None,
        },
        "qubit_count": (models.get(q_best_key, {})
                        .get("resources") or {}).get("n_qubits"),
        "circuit_depth": (models.get(q_best_key, {})
                          .get("resources") or {}).get("circuit_depth"),
        "training_runtime": {
            k: models[k]["train_time_s"]["total"] for k in models},
        "inference_runtime": {
            k: models[k]["model_meta"].get("inference_time_s")
            for k in models},
        "quantum_resource_data": (models.get(q_best_key, {})
                                  .get("resources") or {}),
        "ablation_result": ablation,
        "dequantization_result": dequant,
        "bottleneck_honesty_meter": honesty,
        "statistical_result": {
            "delong_test": delong,
            "corrected_cv_test": cv_t,
            "bootstrap": {k: models[k].get("bootstrap_auroc")
                          for k in models},
        },
        "final_evidence_classification": verdict["classification"],
        "final_evidence_rationale": verdict["rationale"],
        "reproducibility": reproducibility,
        "methodology_notes": [
            "Classification rules are deterministic and documented in code; "
            "no human or model judgement is applied to the verdict.",
            "Statistical significance is only reported when the test was "
            "actually performed (DeLong paired test on pooled out-of-fold "
            "predictions; Nadeau-Bengio corrected resampled t-test).",
            "Exact statevector simulation (equivalent to shots=inf) is used "
            "for quantum circuits; resource reports record this explicitly.",
        ],
    }
    return cert


def certificate_to_markdown(cert: Dict[str, Any]) -> str:
    """Render the certificate as a Markdown report."""
    m = cert["metrics"]
    lines = [
        f"# {cert['certificate_title']}",
        "",
        f"**Experiment:** `{cert['experiment_id']}`  ",
        f"**Dataset:** {cert['dataset']}  ",
        f"**Quantum model:** {cert['model']}  ",
        f"**Configuration hash:** `{cert['configuration_hash']}`  ",
        f"**Seeds:** {cert['random_seeds']}",
        "",
        "## Verdict",
        "",
        f"**{cert['final_evidence_classification']}** — "
        f"{cert['final_evidence_rationale']}",
        "",
        "## Performance (mean AUROC over folds/seeds)",
        "",
        "| Model | Family | AUROC | 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for k, v in sorted(m["all_models"].items(),
                       key=lambda kv: -(kv[1] or 0)):
        ci = cert["confidence_intervals"].get(k)
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci and ci[0] is not None else "—"
        fam = "quantum" if k in ("qsvm", "vqc", "reupload", "hybrid_qnn") else "classical"
        lines.append(f"| {k} | {fam} | {v:.4f} | {ci_str} |")
    lines += [
        "",
        "## Statistical evidence",
        "",
        "```json",
        __import__("json").dumps(cert["statistical_result"], indent=2,
                                 default=str)[:1500],
        "```",
        "",
        "## Entanglement ablation",
        "",
        "```json",
        __import__("json").dumps(cert.get("ablation_result") or {},
                                 indent=2, default=str)[:1200],
        "```",
        "",
        "## Dequantization",
        "",
        "```json",
        __import__("json").dumps(cert.get("dequantization_result") or {},
                                 indent=2, default=str)[:1200],
        "```",
        "",
        "## Bottleneck honesty meter",
        "",
        "```json",
        __import__("json").dumps(cert.get("bottleneck_honesty_meter") or {},
                                 indent=2, default=str)[:800],
        "```",
        "",
        "## Resources",
        "",
        f"- Qubits: {cert.get('qubit_count')}  ",
        f"- Circuit depth: {cert.get('circuit_depth')}  ",
        f"- Quantum parameters: {cert['parameter_count']['quantum']}  ",
        f"- Matched MLP parameters: {cert['parameter_count']['matched_mlp']}",
        "",
        "## Reproducibility",
        "",
        "```json",
        __import__("json").dumps(cert["reproducibility"], indent=2,
                                 default=str)[:1000],
        "```",
        "",
        "---",
        "",
        cert["research_status"],
        "",
    ]
    return "\n".join(lines)
