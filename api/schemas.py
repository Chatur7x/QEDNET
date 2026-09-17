"""API schemas (Pydantic models)."""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DatasetName(str, Enum):
    breast_cancer = "breast_cancer"
    heart_disease = "heart_disease"
    parkinsons = "parkinsons"
    diabetes = "diabetes"


class ModelType(str, Enum):
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    xgboost = "xgboost"
    rbf_svm = "rbf_svm"
    parameter_matched_mlp = "parameter_matched_mlp"
    vqc = "vqc"
    qsvm = "qsvm"
    reupload = "reupload"
    hybrid_qnn = "hybrid_qnn"
    quantum_cascade = "quantum_cascade"


class PreprocessConfig(BaseModel):
    impute_strategy: str = "median"
    scale_method: str = "standard"
    select_method: str = "mutual_info"
    select_k: int = 16
    variance_threshold: float = 0.0
    compress_method: str = "pca"
    compress_dim: int = 8
    ae_hidden_dims: List[int] = [64, 32]
    ae_epochs: int = 50
    smote: bool = False
    smote_k: int = 5
    random_state: int = 42


class ModelConfig(BaseModel):
    type: ModelType
    n_qubits: int = 8
    n_layers: int = 3
    encoding: str = "angle"
    ansatz: str = "strongly_entangling"
    optimizer: str = "adam"
    learning_rate: float = 0.01
    max_iter: int = 50
    shots: Optional[int] = None
    diff_method: str = "backprop"
    device: str = "default.qubit"
    use_entanglement: bool = True
    # Classical
    n_estimators: int = 300
    max_depth: int = 4
    C: float = 1.0
    n_params_target: int = 100
    hidden_dims: List[int] = [32, 16]
    epochs: int = 100


class TrainingConfig(BaseModel):
    outer_folds: int = 5
    inner_folds: int = 3
    seeds: List[int] = [42]
    shots: int = 1024


class EvaluationConfig(BaseModel):
    metrics: List[str] = ["auroc", "auprc", "sens_at_spec95", "mcc", "brier", "ece"]
    bootstrap_ci: int = 2000
    delong_test: bool = True
    decision_curve: bool = True


class CertificateConfig(BaseModel):
    dequantization_attack: List[str] = ["rff", "nystrom"]
    entanglement_ablation: str = "parameter_matched"
    bottleneck_honesty_meter: bool = True


class ExperimentConfig(BaseModel):
    dataset: DatasetName = DatasetName.breast_cancer
    preprocess: PreprocessConfig = PreprocessConfig()
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    certificate: CertificateConfig = CertificateConfig()
    seed: int = 42
    mlflow_uri: str = "file:./mlruns"
    experiment_name: str = "qednet"


class PredictionRequest(BaseModel):
    features: List[float]
    model_id: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    uncertainty: Optional[Dict[str, float]] = None
    model_id: str


class ExperimentRequest(BaseModel):
    config: ExperimentConfig


class ExperimentResponse(BaseModel):
    run_id: str
    status: str
    metrics: Optional[Dict[str, Any]] = None
    circuit_info: Optional[Dict[str, Any]] = None
    certificate: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    available_datasets: List[str]
    available_models: List[str]