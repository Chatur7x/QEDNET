"""Experiment configuration loading, validation and hashing.

Experiments are configured through YAML files (never hard-coded settings).
Every experiment stores its configuration and a deterministic hash so results
are traceable and reproducible.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str = "breast_cancer"
    target: str = "target"
    source: str = "builtin"  # builtin | csv


@dataclass
class CompressionConfig:
    method: str = "pca"  # pca | mutual_info | lasso | identity
    target_dim: int = 8


@dataclass
class ModelConfig:
    type: str = "vqc"            # model key, see models/registry.py
    encoding: str = "angle"      # angle | zz | amplitude | reupload
    n_qubits: int = 8
    layers: int = 3
    backend: str = "numpy"       # numpy (batched statevector) | pennylane
    entanglement: bool = True    # False => parameter-matched ablation circuit
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 7, 2026])
    shots: Optional[int] = None  # None => exact statevector (shots=inf)


@dataclass
class EvaluationConfig:
    bootstrap_samples: int = 2000
    bootstrap_enabled: bool = True
    delong_test: bool = True
    decision_curve: bool = True
    calibration: str = "platt"  # platt | isotonic | none


@dataclass
class CascadeConfig:
    enabled: bool = True
    screening_model: str = "logistic_regression"
    low_confidence_min: float = 0.35
    low_confidence_max: float = 0.65


@dataclass
class CertificateConfig:
    enabled: bool = True
    dequantization_rff: bool = True
    dequantization_nystrom: bool = True
    entanglement_ablation: bool = True
    bottleneck_honesty_meter: bool = True


@dataclass
class ExperimentConfig:
    experiment_id: str = "exp_000"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    certificate: CertificateConfig = field(default_factory=CertificateConfig)
    models: List[str] = field(default_factory=list)  # benchmark suite override
    notes: str = ""

    def config_hash(self) -> str:
        """Deterministic short hash of the full configuration."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _coerce(dc_cls, raw: Dict[str, Any] | None):
    """Build a dataclass instance ignoring unknown keys (forward compatible)."""
    raw = raw or {}
    valid = {f for f in dc_cls.__dataclass_fields__}  # noqa: C416
    return dc_cls(**{k: v for k, v in raw.items() if k in valid})


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = ExperimentConfig(
        experiment_id=str(raw.get("experiment_id", "exp_000")),
        dataset=_coerce(DatasetConfig, raw.get("dataset")),
        compression=_coerce(CompressionConfig, raw.get("compression")),
        model=_coerce(ModelConfig, raw.get("model")),
        training=_coerce(TrainingConfig, raw.get("training")),
        evaluation=_coerce(EvaluationConfig, raw.get("evaluation")),
        cascade=_coerce(CascadeConfig, raw.get("cascade")),
        certificate=_coerce(CertificateConfig, raw.get("certificate")),
        models=list(raw.get("models", [])),
        notes=str(raw.get("notes", "")),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: ExperimentConfig) -> None:
    if cfg.compression.target_dim < 2:
        raise ValueError("compression.target_dim must be >= 2")
    if cfg.training.outer_folds < 2:
        raise ValueError("training.outer_folds must be >= 2")
    if cfg.training.inner_folds < 2:
        raise ValueError("training.inner_folds must be >= 2")
    if not cfg.training.seeds:
        raise ValueError("training.seeds must be a non-empty list")
    if cfg.model.backend not in ("numpy", "pennylane"):
        raise ValueError("model.backend must be 'numpy' or 'pennylane'")
    if cfg.model.n_qubits < 1 or cfg.model.n_qubits > 12:
        raise ValueError("model.n_qubits must be in [1, 12] (resource limits)")
    if cfg.evaluation.calibration not in ("platt", "isotonic", "none"):
        raise ValueError("evaluation.calibration must be platt|isotonic|none")
    lo, hi = cfg.cascade.low_confidence_min, cfg.cascade.low_confidence_max
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("cascade band must satisfy 0 <= min < max <= 1")


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    return asdict(cfg)
