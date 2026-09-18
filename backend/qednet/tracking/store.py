"""Experiment tracking (MLflow-compatible fields, file-based store).

Every experiment records: dataset, model, preprocessing, seed, fold,
hyperparameters, metrics, training/inference time, parameters, qubits,
circuit depth, shots, artifacts, configuration hash. Results are written as
JSON (portable, machine-readable). If MLflow is installed, the same fields
are ALSO logged to a local MLflow tracking store; absence of MLflow never
blocks the experiment.
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("qednet.tracking")

RUNS_DIR = Path("experiments/runs")


def reproducibility_snapshot(config_hash: str) -> Dict[str, Any]:
    """Everything needed to reproduce an experiment."""
    import numpy as np
    import sklearn
    pkgs = {"numpy": np.__version__, "scikit-learn": sklearn.__version__}
    try:
        import pennylane
        pkgs["pennylane"] = pennylane.__version__
    except ImportError:
        pkgs["pennylane"] = "not installed"
    try:
        import xgboost
        pkgs["xgboost"] = xgboost.__version__
    except ImportError:
        pkgs["xgboost"] = "not installed"
    try:
        import shap
        pkgs["shap"] = shap.__version__
    except ImportError:
        pkgs["shap"] = "not installed"
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": pkgs,
        "qednet_version": _qednet_version(),
        "config_hash": config_hash,
        "created_unix": int(time.time()),
    }


def _qednet_version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


class ExperimentStore:
    """File-backed experiment store with optional MLflow mirroring."""

    def __init__(self, runs_dir: Path | None = None,
                 use_mlflow: bool = True):
        self.runs_dir = Path(runs_dir or RUNS_DIR)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._mlflow = None
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(
                    (self.runs_dir / "mlruns").as_uri())
                self._mlflow = mlflow
            except Exception as exc:  # noqa: BLE001
                logger.info("MLflow not available (%s); JSON store only", exc)

    # ------------------------------------------------------------- paths
    def experiment_dir(self, experiment_id: str) -> Path:
        d = self.runs_dir / experiment_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_result(self, experiment_id: str, result: Dict[str, Any]) -> Path:
        path = self.experiment_dir(experiment_id) / "result.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        return path

    def save_json(self, experiment_id: str, name: str,
                  payload: Dict[str, Any]) -> Path:
        path = self.experiment_dir(experiment_id) / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def save_text(self, experiment_id: str, name: str,
                  text: str) -> Path:
        path = self.experiment_dir(experiment_id) / f"{name}.md"
        path.write_text(text)
        return path

    def save_binary(self, experiment_id: str, name: str,
                    obj: Any) -> Path:
        import pickle
        path = self.experiment_dir(experiment_id) / f"{name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        return path

    def load_binary(self, experiment_id: str, name: str) -> Any:
        import pickle
        path = self.runs_dir / experiment_id / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No artifact '{name}' for {experiment_id}")
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def load_json(self, experiment_id: str, name: str) -> Dict[str, Any]:
        path = self.runs_dir / experiment_id / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"No '{name}.json' for {experiment_id}")
        return json.loads(path.read_text())

    # ------------------------------------------------------------- listing
    def list_experiments(self) -> List[Dict[str, Any]]:
        out = []
        for d in sorted(self.runs_dir.iterdir()):
            if not d.is_dir():
                continue
            result_p = d / "result.json"
            cert_p = d / "certificate.json"
            cfg_p = d / "config.yaml"
            if not result_p.exists():
                continue
            try:
                result = json.loads(result_p.read_text())
                entry = {
                    "experiment_id": d.name,
                    "dataset": result.get("dataset"),
                    "n_samples": result.get("n_samples"),
                    "n_features": result.get("n_features"),
                    "models": list(result.get("models", {}).keys()),
                    "protocol": result.get("protocol"),
                    "runtime_s": result.get("runtime_s"),
                    "created": d.stat().st_mtime,
                    "has_certificate": cert_p.exists(),
                    "config_file": str(cfg_p) if cfg_p.exists() else None,
                    "validation_passed": result.get("validation", {}).get("passed"),
                }
                if cert_p.exists():
                    try:
                        cert = json.loads(cert_p.read_text())
                        entry["evidence_classification"] = cert.get(
                            "final_evidence_classification")
                    except json.JSONDecodeError:
                        pass
                out.append(entry)
            except json.JSONDecodeError:
                continue
        return out

    def get_result(self, experiment_id: str) -> Dict[str, Any]:
        return self.load_json(experiment_id, "result")

    # ------------------------------------------------------------- mlflow
    def log_to_mlflow(self, experiment_id: str, cfg, result: Dict[str, Any],
                      artifacts: Optional[List[Path]] = None) -> bool:
        """Mirror the experiment into MLflow if it is installed."""
        if self._mlflow is None:
            return False
        try:
            mlflow = self._mlflow
            mlflow.set_experiment("qednet-2.0")
            with mlflow.start_run(run_name=experiment_id):
                mlflow.set_tag("experiment_id", experiment_id)
                mlflow.set_tag("dataset", result.get("dataset"))
                mlflow.set_tag("config_hash", cfg.config_hash())
                mlflow.log_param("dataset", cfg.dataset.name)
                mlflow.log_param("compression", cfg.compression.method)
                mlflow.log_param("target_dim", cfg.compression.target_dim)
                mlflow.log_param("outer_folds", cfg.training.outer_folds)
                mlflow.log_param("inner_folds", cfg.training.inner_folds)
                mlflow.log_param("seeds", cfg.training.seeds)
                for model_key, m in result.get("models", {}).items():
                    for metric in ("auroc", "auprc", "mcc", "brier"):
                        v = m["summary"][metric]["mean"]
                        if v is not None:
                            mlflow.log_metric(f"{model_key}.{metric}",
                                              round(float(v), 6))
                    pt = m.get("train_time_s", {}).get("total")
                    if pt is not None:
                        mlflow.log_metric(f"{model_key}.train_time_s",
                                          round(float(pt), 3))
                for art in artifacts or []:
                    if Path(art).exists():
                        mlflow.log_artifact(str(art))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging failed (continuing): %s", exc)
            return False
