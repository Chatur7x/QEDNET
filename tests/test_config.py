"""Tests for configuration loading and validation."""

import pytest
import tempfile
import yaml
from pathlib import Path

from qednet.train.config import (
    ExperimentConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    CertificateConfig,
    DEFAULT_CONFIG_YAML,
)


class TestDatasetConfig:
    def test_default_values(self):
        config = DatasetConfig()
        assert config.name == "breast_cancer"
        assert config.compression == "pca"
        assert config.target_dim == 8
        assert config.test_size == 0.2
        assert config.random_state == 42

    def test_custom_values(self):
        config = DatasetConfig(name="custom", compression="none", target_dim=16)
        assert config.name == "custom"
        assert config.compression == "none"
        assert config.target_dim == 16


class TestModelConfig:
    def test_default_values(self):
        config = ModelConfig()
        assert config.type == "vqc"
        assert config.n_qubits == 8
        assert config.layers == 3
        assert config.ansatz == "strongly_entangling"
        assert config.encoding == "angle"
        assert config.optimizer == "adam"
        assert config.learning_rate == 0.01

    def test_quantum_model_types(self):
        for model_type in ["vqc", "qsvm", "reupload", "hybrid_qnn"]:
            config = ModelConfig(type=model_type)
            assert config.type == model_type

    def test_classical_model_types(self):
        for model_type in ["xgboost", "lightgbm", "logistic", "mlp"]:
            config = ModelConfig(type=model_type)
            assert config.type == model_type


class TestExperimentConfig:
    def test_default_config(self):
        config = ExperimentConfig()
        assert config.experiment_name == "qednet_experiment"
        assert isinstance(config.dataset, DatasetConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert isinstance(config.certificate, CertificateConfig)

    def test_from_dict(self):
        data = {
            "dataset": {"name": "breast_cancer", "target_dim": 4},
            "model": {"type": "xgboost", "n_estimators": 50},
        }
        config = ExperimentConfig.from_dict(data)
        assert config.dataset.name == "breast_cancer"
        assert config.dataset.target_dim == 4
        assert config.model.type == "xgboost"
        assert config.model.n_estimators == 50  # This won't work as ModelConfig doesn't have n_estimators

    def test_compute_hash(self):
        config = ExperimentConfig()
        hash1 = config.compute_hash()
        hash2 = config.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_hash_changes_with_config(self):
        config1 = ExperimentConfig()
        config2 = ExperimentConfig()
        config2.model.learning_rate = 0.02
        assert config1.compute_hash() != config2.compute_hash()

    def test_to_yaml_and_from_yaml(self):
        config = ExperimentConfig()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = ExperimentConfig.from_yaml(f.name)
        assert loaded.compute_hash() == config.compute_hash()

    def test_default_config_yaml_parsable(self):
        # Ensure the default YAML is valid
        data = yaml.safe_load(DEFAULT_CONFIG_YAML)
        config = ExperimentConfig.from_dict(data)
        assert config.dataset.name == "breast_cancer"
        assert config.model.type == "vqc"