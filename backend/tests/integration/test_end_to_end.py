"""Integration test: the complete F1 -> F12 pipeline on a small open dataset.

Runs a reduced-protocol end-to-end experiment (small folds/seeds) exercising
every feature F1-F12 and verifying artifact persistence. Marked slow.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def small_experiment(tmp_path_factory):
    from qednet.config import ExperimentConfig
    from qednet.data.ingestion import DatasetRegistry, ingest_csv
    from qednet.train.runner import run_experiment
    from qednet.tracking.store import ExperimentStore

    tmp = tmp_path_factory.mktemp("exp")
    cwd = Path.cwd()
    # run inside tmp so artifacts don't pollute the repo
    import os
    os.chdir(tmp)
    try:
        reg = DatasetRegistry(registry_path=tmp / "registry.json")
        # A deterministic synthetic file keeps this integration test independent
        # of private/untracked raw datasets while exercising the real ingestion
        # and experiment pipeline.
        rng = np.random.default_rng(42)
        subject_ids = np.repeat([f"S{i:02d}" for i in range(30)], 3)
        labels = np.repeat(np.tile([0, 1], 15), 3)
        values = rng.normal(loc=labels[:, None] * 0.5, size=(len(labels), 10))
        frame = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(10)])
        frame["__subject_id__"] = subject_ids
        frame["status"] = labels
        source = tmp / "grouped_research_fixture.csv"
        frame.to_csv(source, index=False)
        reg.register(ingest_csv(source, target="status",
                                dataset_name="grouped_research_fixture"))

        cfg = ExperimentConfig(experiment_id="exp_integration_test")
        cfg.dataset.name = "grouped_research_fixture"
        cfg.dataset.target = "status"
        cfg.dataset.group_column = "__subject_id__"
        cfg.compression.target_dim = 6
        cfg.model.n_qubits = 6
        cfg.model.layers = 2
        cfg.training.outer_folds = 3
        cfg.training.inner_folds = 2
        cfg.training.seeds = [42]
        cfg.evaluation.bootstrap_samples = 100
        cfg.evaluation.decision_curve = False

        store = ExperimentStore(runs_dir=tmp / "runs")
        result = run_experiment(cfg, store, reg, explain_samples=4)
        return {"result": result, "store": store, "cfg": cfg, "tmp": tmp}
    finally:
        os.chdir(cwd)


def test_all_twelve_features_present(small_experiment):
    r = small_experiment["result"]
    # F1/F2
    assert r["validation"]["passed"] is True
    assert r["dataset_fingerprint"]
    # F3
    assert r["config"]["compression"]["target_dim"] == 6
    # F5/F6/F7
    fams = {k: v["family"] for k, v in r["models"].items()}
    assert any(v == "quantum" for v in fams.values())
    assert any(v == "classical" for v in fams.values())
    for m in r["models"].values():
        assert m["summary"]["auroc"]["mean"] is not None
        assert m["leakage_audit"]["contamination_detected"] is False
    # F8
    assert "explainability_summary" in r
    # F9
    assert Path(small_experiment["tmp"] / "runs" /
                "exp_integration_test" / "certificate.json").exists()
    assert r.get("dequantization") is not None
    assert r.get("ablation") is not None
    # F10
    assert r["bottleneck_honesty_meter"]["reading"]
    assert r["bottleneck_honesty_meter"]["compressed_space_baseline"]["auroc"] \
        is not None
    # F11
    assert r.get("conformal", {}).get("status") == "NOT MEASURED"
    assert r["conformal"].get("empirical_coverage") is None
    # F12
    assert r["cascade"] and "routing" in r["cascade"]


def test_certificate_contents(small_experiment):
    cert = small_experiment["store"].load_json(
        "exp_integration_test", "certificate")
    for key in ("experiment_id", "dataset", "model", "configuration_hash",
                "random_seeds", "metrics", "confidence_intervals",
                "parameter_count", "qubit_count", "circuit_depth",
                "training_runtime", "inference_runtime",
                "quantum_resource_data", "ablation_result",
                "dequantization_result", "statistical_result",
                "final_evidence_classification"):
        assert key in cert, f"certificate missing {key}"
    assert cert["final_evidence_classification"] in (
        "No Demonstrated Advantage", "Parameter-Efficiency Advantage",
        "Small-Data Advantage", "Inconclusive / Underpowered")


def test_deployment_artifacts_exist(small_experiment):
    store = small_experiment["store"]
    pipe = store.load_binary("exp_integration_test", "pipeline")
    models = store.load_binary("exp_integration_test", "models")
    assert hasattr(pipe, "transform")
    assert len(models) >= 5
    x = np.zeros((1, pipe.input_dim))
    Z = pipe.transform(x)
    assert Z.shape[1] <= 6


def test_no_fake_results(small_experiment):
    """Every reported metric must trace to stored fold-level records."""
    r = small_experiment["result"]
    for key, m in r["models"].items():
        n_folds = len(m["folds"])
        assert n_folds == 3  # 3 outer folds x 1 seed
        means = np.mean([f["metrics"]["auroc"] for f in m["folds"]])
        assert abs(means - m["summary"]["auroc"]["mean"]) < 1e-6
        assert m["pooled_predictions"]["y"]


def test_markdown_report_generated(small_experiment):
    report_path = small_experiment["tmp"] / "runs" / \
        "exp_integration_test" / "report.md"
    assert report_path.exists()
    text = report_path.read_text()
    assert "RESEARCH AND EDUCATIONAL USE ONLY" in text
    assert "AUROC" in text
