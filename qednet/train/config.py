"""D7 Training pipeline with YAML configs."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import yaml

from qednet.data import DatasetRegistry
from qednet.preprocess import make_preprocess_pipeline, PreprocessConfig
from qednet.models import (
    VQC, QSVM, ReuploadClassifier, HybridQNNLayer, make_hybrid_model,
    LogisticRegressionModel, RandomForestModel, XGBoostModel,
    RBFSVMModel, ParameterMatchedMLP,
)


@dataclass
class ModelConfig:
    type: str = "vqc"
    n_qubits: int = 8
    n_layers: int = 3
    encoding: str = "angle"
    ansatz: str = "strongly_entangling"
    optimizer: str = "adam"
    learning_rate: float = 0.01
    max_iter: int = 50
    shots: int | None = None
    diff_method: str = "backprop"
    device: str = "default.qubit"
    use_entanglement: bool = True
    # Classical model params
    n_estimators: int = 300
    max_depth: int = 4
    C: float = 1.0
    n_params_target: int = 100
    hidden_dims: tuple = (32, 16)
    epochs: int = 100


@dataclass
class DatasetConfig:
    name: str = "breast_cancer"
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class PreprocessCfg:
    impute_strategy: str = "median"
    scale_method: str = "standard"
    select_method: str = "mutual_info"
    select_k: int = 16
    variance_threshold: float = 0.0
    compress_method: str = "pca"
    compress_dim: int = 8
    ae_hidden_dims: tuple = (64, 32)
    ae_epochs: int = 50
    smote: bool = False
    smote_k: int = 5
    random_state: int = 42


@dataclass
class TrainingConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    seeds: list[int] = field(default_factory=lambda: [42])
    shots: int = 1024


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: ["auroc", "auprc", "sens_at_spec95", "mcc", "brier", "ece"])
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
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    certificate: CertificateConfig = field(default_factory=CertificateConfig)
    seed: int = 42
    mlflow_uri: str = "file:./mlruns"
    experiment_name: str = "qednet"


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return ExperimentConfig(
        dataset=DatasetConfig(**cfg.get("dataset", {})),
        preprocess=PreprocessCfg(**cfg.get("preprocess", {})),
        model=ModelConfig(**cfg.get("model", {})),
        training=TrainingConfig(**cfg.get("training", {})),
        evaluation=EvaluationConfig(**cfg.get("evaluation", {})),
        certificate=CertificateConfig(**cfg.get("certificate", {})),
        seed=cfg.get("seed", 42),
        mlflow_uri=cfg.get("mlflow_uri", "file:./mlruns"),
        experiment_name=cfg.get("experiment_name", "qednet"),
    )


def save_config(config: ExperimentConfig, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(asdict(config), f, default_flow_style=False)


def create_model(model_cfg: ModelConfig, n_qubits: int, n_features: int, seed: int = 42):
    """Factory for creating models from config."""
    mtype = model_cfg.type.lower()
    common = dict(random_state=seed)

    if mtype == "vqc":
        return VQC(
            n_qubits=n_qubits,
            n_layers=model_cfg.n_layers,
            encoding=model_cfg.encoding,
            device=model_cfg.device,
            shots=model_cfg.shots,
            diff_method=model_cfg.diff_method,
            optimizer=model_cfg.optimizer,
            lr=model_cfg.learning_rate,
            max_iter=model_cfg.max_iter,
            seed=seed,
            use_entanglement=model_cfg.use_entanglement,
        )
    elif mtype == "qsvm":
        return QSVM(
            n_qubits=n_qubits,
            encoding=model_cfg.encoding,
            device=model_cfg.device,
            shots=model_cfg.shots,
            C=model_cfg.C,
        )
    elif mtype == "reupload":
        return ReuploadClassifier(
            n_qubits=n_qubits,
            n_layers=model_cfg.n_layers,
            device=model_cfg.device,
            shots=model_cfg.shots,
            diff_method=model_cfg.diff_method,
            optimizer=model_cfg.optimizer,
            lr=model_cfg.learning_rate,
            max_iter=model_cfg.max_iter,
            seed=seed,
        )
    elif mtype == "hybrid_qnn":
        return make_hybrid_model(
            input_dim=n_features,
            n_qubits=n_qubits,
            n_layers=model_cfg.n_layers,
            encoding=model_cfg.encoding,
            device=model_cfg.device,
            diff_method=model_cfg.diff_method,
            seed=seed,
        )
    elif mtype == "logistic_regression":
        return LogisticRegressionModel(C=model_cfg.C, **common)
    elif mtype == "random_forest":
        return RandomForestModel(n_estimators=model_cfg.n_estimators, max_depth=model_cfg.max_depth, **common)
    elif mtype == "xgboost":
        return XGBoostModel(
            n_estimators=model_cfg.n_estimators,
            max_depth=model_cfg.max_depth,
            learning_rate=model_cfg.learning_rate,
            **common,
        )
    elif mtype == "rbf_svm":
        return RBFSVMModel(C=model_cfg.C, **common)
    elif mtype == "parameter_matched_mlp":
        return ParameterMatchedMLP(
            n_params_target=model_cfg.n_params_target,
            input_dim=n_features,
            hidden_dims=model_cfg.hidden_dims,
            epochs=model_cfg.epochs,
            **common,
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def run_experiment(config: ExperimentConfig, model_override: str | None = None):
    """Run a complete experiment with nested CV."""
    try:
        import mlflow
    except ImportError:
        mlflow = None  # ponytail: tracking optional; results still returned
    if mlflow is not None:
        mlflow.set_tracking_uri(config.mlflow_uri)
        mlflow.set_experiment(config.experiment_name)

    # Load data
    bundle = DatasetRegistry.load(config.dataset.name, **config.dataset.__dict__)
    X, y = bundle.X, bundle.y

    # Preprocess
    preprocess = make_preprocess_pipeline(PreprocessConfig(**config.preprocess.__dict__))

    # Determine quantum input dimension
    n_features_compressed = config.preprocess.compress_dim

    from contextlib import nullcontext
    run_ctx = mlflow.start_run() if mlflow is not None else nullcontext()
    with run_ctx:
        if mlflow is not None:
            mlflow.log_params({f"config_{k}": v for k, v in asdict(config).items() if not isinstance(v, dict)})
            for section, vals in asdict(config).items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        mlflow.log_param(f"{section}.{k}", v)

        model_type = model_override or config.model.type
        if mlflow is not None:
            mlflow.log_param("model_type", model_type)

        # For now just fit on full data (nested CV in evaluation module)
        model = create_model(config.model, n_features_compressed, X.shape[1], config.seed)
        X_trans = preprocess.fit_transform(X, y)
        model.fit(X_trans, y)

        if mlflow is not None:
            mlflow.log_metric("train_samples", len(X_trans))
            mlflow.log_metric("n_features_in", X.shape[1])
            mlflow.log_metric("n_features_compressed", X_trans.shape[1])

            if hasattr(model, "metadata"):
                for k, v in model.metadata().items():
                    mlflow.log_metric(f"model_{k}", v) if isinstance(v, (int, float)) else mlflow.log_param(f"model_{k}", str(v))

        return model, preprocess, bundle