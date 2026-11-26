import json
from types import SimpleNamespace

import pandas as pd

from qednet import cli
from qednet.data.ingestion import (
    DatasetRegistry,
    RegisteredDataset,
    profile_frame,
)


def test_profile_excludes_group_identifier_from_predictive_features():
    frame = pd.DataFrame({
        "signal": [0.1, 0.2, 0.3, 0.4],
        "__subject_id__": ["S1", "S1", "S2", "S2"],
        "status": [0, 1, 0, 1],
    })

    profile = profile_frame(frame, "status", group_column="__subject_id__")

    assert profile.n_features == 1
    assert profile.feature_names == ["signal"]
    assert profile.identifier_like_columns == ["__subject_id__"]


def test_parkinsons_validation_excludes_subject_group_from_features(
    monkeypatch, capsys
):
    frame = pd.DataFrame({
        "signal": [0.1, 0.2, 0.3, 0.4],
        "__subject_id__": ["S1", "S1", "S2", "S2"],
        "status": [0, 1, 0, 1],
    })

    class Registry:
        validation = None

        def get(self, name):
            assert name == "parkinsons"
            return {"target": "status"}

        def update_validation(self, name, report):
            assert name == "parkinsons"
            self.validation = report

    registry = Registry()
    monkeypatch.setattr(cli, "_registry", lambda: registry)
    monkeypatch.setattr(
        "qednet.data.datasets.load_dataset",
        lambda name, registry: frame,
    )

    cli.cmd_validate(SimpleNamespace(dataset="parkinsons"))
    report = json.loads(capsys.readouterr().out)

    assert report["passed"] is True
    assert report["stats"]["group_column"] == "__subject_id__"
    assert registry.validation == report


def test_validation_status_and_report_persist_across_registry_instances(tmp_path):
    path = tmp_path / "datasets.json"
    registry = DatasetRegistry(registry_path=path)
    registry.register(RegisteredDataset(name="demo", source="csv", target="y"))

    report = {"dataset": "demo", "passed": False, "errors": ["bad target"]}
    registry.update_validation("demo", report)

    restored = DatasetRegistry(registry_path=path).get("demo")
    assert restored["validation_status"] == "failed"
    assert restored["validation_report"] == report
    assert isinstance(restored["validated_at"], str)
