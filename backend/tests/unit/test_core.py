"""Unit tests: models, metrics, calibration, uncertainty, certificate."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qednet.evaluation.metrics import (compute_metrics,
                                       expected_calibration_error,
                                       sensitivity_at_specificity)
from qednet.evaluation.statistics import (bootstrap_ci, corrected_cv_ttest,
                                          delong_test, decision_curve_data)
from qednet.models.registry import (CLASSICAL_MODELS, QUANTUM_MODELS,
                                    build_model)
from qednet.uncertainty.abstention import (AbstentionRule, ConformalClassifier,
                                           coverage_stats, predictive_entropy)


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------- metrics
def test_metrics_perfect_classifier():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.001, 0.001, 0.001, 0.999, 0.999, 0.999])
    m = compute_metrics(y, p)
    assert m["auroc"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["brier"] < 0.01
    assert m["ece"] < 0.01
    # confidence (0.95) below accuracy (1.0) must be measurable as ECE
    m2 = compute_metrics(y, np.array([0.05, 0.05, 0.05, 0.95, 0.95, 0.95]))
    assert abs(m2["ece"] - 0.05) < 1e-9


def test_metrics_random_classifier():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    p = rng.uniform(0, 1, 400)
    m = compute_metrics(y, p)
    assert abs(m["auroc"] - 0.5) < 0.1
    assert m["brier"] > 0.2


def test_sens_at_spec_95_operating_point():
    y = np.array([0] * 100 + [1] * 100)
    p = np.concatenate([np.linspace(0.01, 0.4, 100),
                        np.linspace(0.6, 0.99, 100)])
    m = compute_metrics(y, p)
    assert m["sens_at_spec_95"] >= 0.9  # separable data reaches high sens


def test_ece_zero_for_perfectly_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.1, 0.9, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.06


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("key", CLASSICAL_MODELS)
def test_classical_models_fit_predict(key):
    X, y = _toy()
    m = build_model(key, random_state=0,
                    **({"match_params": 50} if key == "matched_mlp" else {}))
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (len(y),)
    assert np.all((p >= 0) & (p <= 1))
    preds = m.predict(X)
    assert set(np.unique(preds)) <= {0, 1}
    assert m.n_parameters() > 0
    assert m.quantum_resources() is None
    meta = m.metadata()
    assert meta["family"] == "classical"


@pytest.mark.parametrize("key", QUANTUM_MODELS)
def test_quantum_models_fit_predict(key):
    X, y = _toy(80)
    kwargs = dict(n_qubits=6, layers=2, epochs=8, batch_size=16)
    m = build_model(key, random_state=0, **kwargs)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (len(y),)
    assert np.all((p >= 0) & (p <= 1))
    assert m.n_parameters() > 0
    res = m.quantum_resources()
    assert res is not None and res["n_qubits"] == 6
    meta = m.metadata()
    assert meta["family"] == "quantum"
    assert res["gate_counts"]  # honest gate accounting present


def test_vqc_learns_separable_data():
    X, y = _toy(160)
    m = build_model("vqc", random_state=0, n_qubits=6, layers=2,
                    epochs=30, batch_size=32)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p) > 0.85  # must actually learn


def test_reupload_learns_separable_data():
    X, y = _toy(160)
    m = build_model("reupload", random_state=0, n_qubits=6, layers=2,
                    epochs=30, batch_size=32)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p) > 0.85


def test_hybrid_qnn_learns_separable_data():
    X, y = _toy(160)
    m = build_model("hybrid_qnn", random_state=0, n_qubits=6, layers=2,
                    epochs=40, batch_size=32)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p) > 0.85


def test_qsvm_kernel_is_real_quantum_kernel():
    X, y = _toy(60)
    m = build_model("qsvm", random_state=0, n_qubits=6, encoding="zz")
    m.fit(X, y)
    K = m.quantum_kernel(X[:10], X[:10])
    assert K.shape == (10, 10)
    assert np.allclose(np.diag(K), 1.0, atol=1e-8)
    assert np.allclose(K, K.T, atol=1e-8)
    p = m.predict_proba(X[:10])[:, 1]
    assert np.all((p >= 0) & (p <= 1))


def test_entanglement_flag_changes_circuit():
    X, y = _toy(60)
    m_on = build_model("vqc", random_state=0, n_qubits=4, layers=2,
                       epochs=2, entanglement=True)
    m_off = build_model("vqc", random_state=0, n_qubits=4, layers=2,
                        epochs=2, entanglement=False)
    m_on.fit(X, y)
    m_off.fit(X, y)
    g_on = m_on.quantum_resources()["gate_counts"].get("cnot", 0)
    g_off = m_off.quantum_resources()["gate_counts"].get("cnot", 0)
    assert g_on > 0 and g_off == 0
    # parameter count identical (CNOTs carry no parameters)
    assert m_on.n_parameters() == m_off.n_parameters()


# ---------------------------------------------------------------- statistics
def test_delong_test_detects_identical_models():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    p = rng.uniform(0, 1, 300)
    d = delong_test(y, p, p)
    assert d["auc_diff"] == 0.0
    assert d["p_value"] > 0.99  # identical => no difference


def test_delong_test_detects_real_difference():
    rng = np.random.default_rng(5)
    y = np.concatenate([np.zeros(150), np.ones(150)])
    p_good = np.concatenate([rng.uniform(0, 0.3, 150),
                             rng.uniform(0.7, 1, 150)])
    p_bad = rng.uniform(0.3, 0.7, 300)
    d = delong_test(y, p_good, p_bad)
    assert d["auc1"] > d["auc2"]
    assert d["p_value"] < 0.05


def test_bootstrap_ci_covers_point():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 200)
    p = 0.3 + 0.4 * y + rng.normal(0, 0.1, 200)
    ci = bootstrap_ci(y, p, "auroc", n_resamples=500)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["ci_high"] > ci["ci_low"]


def test_corrected_cv_ttest_runs():
    a = [0.91, 0.90, 0.92, 0.89, 0.91]
    b = [0.85, 0.84, 0.86, 0.83, 0.85]
    t = corrected_cv_ttest(a, b, n_train=400, n_test=100, k_folds=5)
    assert t["available"]
    assert t["mean_diff"] == pytest.approx(np.mean(a) - np.mean(b))


def test_decision_curve_renders():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.full(50, 0.2), np.full(50, 0.8)])
    d = decision_curve_data(y, p)
    assert len(d["thresholds"]) == len(d["model"])
    assert all(v >= -0.5 for v in d["model"])


# ---------------------------------------------------------------- uncertainty
def test_conformal_coverage_near_target():
    rng = np.random.default_rng(8)
    n = 2000
    y = rng.integers(0, 2, n)
    p = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, n), 0.01, 0.99)
    P = np.column_stack([1 - p, p])
    conf = ConformalClassifier(alpha=0.1)
    conf.calibrate(y[:1200], P[:1200])
    sets = conf.predict_set(P[1200:])
    stats = coverage_stats(sets, y[1200:])
    assert abs(stats["empirical_coverage"] - 0.9) < 0.06


def test_abstention_rule_blocks_ambiguous():
    rule = AbstentionRule(min_confidence=0.5, band_low=0.35, band_high=0.65)
    d = rule.decide(np.array([0.9, 0.5, 0.2]))
    assert d["abstain"][0] is False or d["abstain"][0] == False  # noqa: E712
    assert bool(d["abstain"][1]) is True   # inside ambiguity band
    assert bool(d["abstain"][2]) is False  # confident negative


def test_predictive_entropy_extrema():
    assert predictive_entropy(np.array([0.5]))[0] == pytest.approx(1.0, abs=1e-6)
    assert predictive_entropy(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- certificate
def test_certificate_classification_rules():
    from qednet.certificate.certificate import classify_evidence
    v = classify_evidence({
        "quantum_best_auroc": 0.90, "classical_best_auroc": 0.92,
        "delong_p": 0.6, "corrected_cv_p": 0.7,
        "quantum_vs_matched_mlp_param_ratio": 0.05, "n_samples": 569})
    assert v["classification"] == "No Demonstrated Advantage"

    v2 = classify_evidence({
        "quantum_best_auroc": 0.95, "classical_best_auroc": 0.85,
        "delong_p": 0.001, "corrected_cv_p": 0.002,
        "quantum_vs_matched_mlp_param_ratio": 0.05, "n_samples": 569})
    assert v2["classification"] == "Parameter-Efficiency Advantage"

    v3 = classify_evidence({
        "quantum_best_auroc": 0.93, "classical_best_auroc": 0.90,
        "delong_p": 0.4, "corrected_cv_p": 0.5,
        "quantum_vs_matched_mlp_param_ratio": 0.3, "n_samples": 195})
    assert v3["classification"] == "Inconclusive / Underpowered"


# ---------------------------------------------------------------- ingestion
def test_ingest_csv_rejects_invalid(tmp_path):
    from qednet.data.ingestion import IngestionError, ingest_csv
    # unsupported extension
    f1 = tmp_path / "data.txt"
    f1.write_text("a,b\n1,2\n")
    with pytest.raises(IngestionError):
        ingest_csv(f1)
    # empty csv
    f2 = tmp_path / "empty.csv"
    f2.write_text("")
    with pytest.raises(IngestionError):
        ingest_csv(f2)
    # missing target
    f3 = tmp_path / "noTarget.csv"
    f3.write_text("a,b\n1,2\n2,3\n")
    with pytest.raises(IngestionError):
        ingest_csv(f3, target="nope")


def test_ingest_csv_profiles_valid_file(tmp_path):
    import pandas as pd
    from qednet.data.ingestion import ingest_csv
    df = pd.DataFrame({
        "f1": np.linspace(0, 1, 50), "f2": np.linspace(1, 2, 50),
        "t": [0] * 25 + [1] * 25})
    f = tmp_path / "good.csv"
    df.to_csv(f, index=False)
    rec = ingest_csv(f, target="t")
    assert rec.n_samples == 50
    assert rec.n_features == 2
    assert rec.profile["class_balance"]["1"] == 0.5


def test_safe_filename_blocks_traversal():
    from qednet.data.ingestion import IngestionError, safe_filename
    assert safe_filename("my dataset v2") == "my_dataset_v2"
    with pytest.raises(IngestionError):
        safe_filename("///..")
