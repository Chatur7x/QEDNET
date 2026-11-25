"""API integration: health -> datasets -> train -> predict -> explain -> certificate."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api.research import app

client = TestClient(app)


def test_health_models_datasets():
    assert client.get("/health").json()["status"] == "ok"
    assert "vqc" in client.get("/models").json()["models"]
    ds = {d["name"]: d for d in client.get("/datasets").json()["datasets"]}
    assert ds["breast_cancer"]["samples"] == 569
    advertised = client.get("/models").json()
    assert "hybrid_qnn" not in advertised["models"]
    assert "quantum_cascade" not in advertised["models"]
    assert set(advertised["quantum"]) == {"vqc", "qsvm", "reupload"}


def test_parameter_matched_mlp_defers_input_width_to_transformed_training_data():
    pytest.importorskip("torch")
    from api.research import create_model
    from api.schemas import ModelConfig

    model = create_model(ModelConfig(type="parameter_matched_mlp"), 4, 30, 42)
    assert model.input_dim is None


def test_train_rejects_unimplemented_compare_model():
    response = client.post(
        "/train",
        json={
            "config": {"model": {"type": "logistic_regression"}},
            "compare": ["quantum_cascade"],
        },
    )
    assert response.status_code == 400


def test_run_and_upload_ids_reject_path_fragments():
    from fastapi import HTTPException

    from api.research import _run_dir

    with pytest.raises(HTTPException) as exc:
        _run_dir("../../outside")
    assert exc.value.status_code == 404
    assert client.get("/experiments/not-valid").status_code == 404
    response = client.post(
        "/datasets/validate",
        json={
            "file_id": "../../runs",
            "target": "target",
        },
    )
    assert response.status_code == 400


def test_upload_enforces_byte_limit_and_removes_partial_file(monkeypatch):
    import api.research as research

    monkeypatch.setattr(research, "MAX_UPLOAD_BYTES", 3)
    before = set(research.UPLOADS.glob("*.csv"))
    response = client.post(
        "/datasets/upload",
        files={"file": ("large.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert response.status_code == 413
    assert set(research.UPLOADS.glob("*.csv")) == before


def test_prediction_rejects_non_finite_features_before_loading_artifacts():
    response = client.post(
        "/predict",
        json={
            "experiment_id": "exp_12345678",
            "features": ["NaN"],
        },
    )
    assert response.status_code == 422


def test_upload_and_validate_csv(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    with open(p, "rb") as f:
        r = client.post("/datasets/upload", files={"file": ("t.csv", f, "text/csv")})
    assert r.status_code == 200
    fid = r.json()["file_id"]
    v = client.post("/datasets/validate", json={"file_id": fid, "target": "target"})
    assert v.json()["valid"] is True


def test_train_predict_explain_certificate():
    r = client.post(
        "/train",
        json={
            "config": {
                "dataset": "breast_cancer",
                "model": {"type": "logistic_regression"},
                "training": {"outer_folds": 2, "seeds": [42]},
                "preprocess": {"select_k": 8, "compress_dim": 4},
            },
            "compare": ["logistic_regression"],
        },
    )
    assert r.status_code == 200
    exp_id = r.json()["experiment_id"]
    for _ in range(120):
        st = client.get(f"/experiments/{exp_id}").json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(2)
    assert st["status"] == "completed", st

    res = client.get(f"/benchmark/{exp_id}").json()
    assert res["models"]["logistic_regression"]["mean"] > 0.9

    feats = res["feature_names"]
    assert len(feats) == 30
    b = __import__("qednet.data", fromlist=["DatasetRegistry"]).DatasetRegistry.load(
        "breast_cancer"
    )
    p = client.post("/predict", json={"experiment_id": exp_id, "features": b.X[0].tolist()})
    assert 0.0 <= p.json()["probability"] <= 1.0
    e = client.post("/explain", json={"experiment_id": exp_id, "features": b.X[0].tolist()})
    assert len(e.json()["sensitivity"]) == 30
    c = client.get(f"/certificate/{exp_id}").json()
    assert c["evidence_classification"] in (
        "No Demonstrated Advantage",
        "Parameter-Efficiency Advantage",
        "Small-Data Advantage",
        "Inconclusive / Underpowered",
    )
