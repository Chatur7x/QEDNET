"""Configuration schema for experiments."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class DatasetConfig:
    name: str = "breast_cancer"
    compression: Literal["pca", "none", "autoencoder"] = "pca"
    target_dim: int = 8
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    type: Literal[
        "vqc", "qsvm", "reupload", "hybrid_qnn", "xgboost", "lightgbm", "logistic", "mlp"
    ] = "vqc"
    n_qubits: int = 8
    layers: int = 3
    ansatz: str = "strongly_entangling"
    encoding: str = "angle"
    optimizer: str = "adam"
    learning_rate: float = 0.01
    max_iter: int = 100
    shots: int = 1024
    random_state: int = 42
    # classical hyperparams (passed through by the factory)
    n_estimators: int = 100
    max_depth: int = 6
    C: float = 1.0


@dataclass
class TrainingConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    seeds: int = 10
    shots: int = 1024
    n_jobs: int = 1
    early_stopping: bool = True
    patience: int = 10


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(
        default_factory=lambda: ["auroc", "auprc", "sens_at_spec95", "mcc", "brier", "ece"]
    )
    bootstrap_ci: int = 2000
    delong_test: bool = True
    decision_curve: bool = True


@dataclass
class CertificateConfig:
    dequantization_attack: list[str] = field(default_factory=lambda: ["rff", "nystrom"])
    entanglement_ablation: str = "parameter_matched"
    bottleneck_honesty_meter: bool = True


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    certificate: CertificateConfig = field(default_factory=CertificateConfig)
    experiment_name: str = "qednet_experiment"
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_mlflow_params(self) -> dict[str, str]:
        """Return stable, flat string parameters suitable for MLflow logging."""
        params: dict[str, str] = {}
        for section, values in self.to_dict().items():
            if isinstance(values, dict):
                for key, value in values.items():
                    rendered = (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else str(value)
                    )
                    params[f"{section}.{key}"] = rendered
            else:
                params[section] = str(values)
        return params

    def compute_hash(self) -> str:
        """Compute deterministic hash of configuration for reproducibility."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(
            dataset=DatasetConfig(**data.get("dataset", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            certificate=CertificateConfig(**data.get("certificate", {})),
            experiment_name=data.get("experiment_name", "qednet_experiment"),
            tags=data.get("tags", {}),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


DEFAULT_CONFIG_YAML = """
dataset:
  name: breast_cancer
  compression: pca
  target_dim: 8
  test_size: 0.2
  random_state: 42

model:
  type: vqc
  n_qubits: 8
  layers: 3
  ansatz: strongly_entangling
  encoding: angle
  optimizer: adam
  learning_rate: 0.01
  max_iter: 100
  shots: 1024
  random_state: 42

training:
  outer_folds: 5
  inner_folds: 3
  seeds: 10
  shots: 1024
  n_jobs: 1
  early_stopping: true
  patience: 10

evaluation:
  metrics:
    - auroc
    - auprc
    - sens_at_spec95
    - mcc
    - brier
    - ece
  bootstrap_ci: 2000
  delong_test: true
  decision_curve: true

certificate:
  dequantization_attack:
    - rff
    - nystrom
  entanglement_ablation: parameter_matched
  bottleneck_honesty_meter: true

experiment_name: qednet_breast_cancer_vqc
tags:
  project: qednet
  disease: breast_cancer
"""
