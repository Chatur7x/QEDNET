"""Tests for model interfaces and factory."""

import pytest
import numpy as np
from numpy.typing import NDArray

from qednet.models.base import BaseEstimator, BaseTransformer
from qednet.models.factory import ModelFactory, create_model
from qednet.models.quantum.mocks import get_mock_model
from qednet.train.config import ModelConfig


class TestBaseEstimator:
    def test_interface_exists(self):
        """Verify BaseEstimator has required methods."""
        assert hasattr(BaseEstimator, "fit")
        assert hasattr(BaseEstimator, "predict_proba")
        assert hasattr(BaseEstimator, "predict")
        assert hasattr(BaseEstimator, "get_params")
        assert hasattr(BaseEstimator, "set_params")

    def test_cannot_instantiate_abstract(self):
        """Cannot instantiate abstract base class directly."""
        with pytest.raises(TypeError):
            BaseEstimator()


class TestBaseTransformer:
    def test_interface_exists(self):
        """Verify BaseTransformer has required methods."""
        assert hasattr(BaseTransformer, "fit")
        assert hasattr(BaseTransformer, "transform")
        assert hasattr(BaseTransformer, "fit_transform")
        assert hasattr(BaseTransformer, "get_params")
        assert hasattr(BaseTransformer, "set_params")

    def test_cannot_instantiate_abstract(self):
        """Cannot instantiate abstract base class directly."""
        with pytest.raises(TypeError):
            BaseTransformer()


class TestMockQuantumModels:
    @pytest.mark.parametrize("model_type", ["vqc", "qsvm", "reupload", "hybrid_qnn"])
    def test_mock_model_creation(self, model_type):
        """Test creating each mock quantum model."""
        model = get_mock_model(model_type, n_qubits=4, max_iter=5, shots=100)
        assert isinstance(model, BaseEstimator)

    @pytest.mark.parametrize("model_type", ["vqc", "qsvm", "reupload", "hybrid_qnn"])
    def test_mock_model_fit_predict(self, model_type, sample_data):
        """Test fit and predict_proba for each mock model."""
        X, y = sample_data
        model = get_mock_model(model_type, n_qubits=4, max_iter=5, shots=100)

        # Fit
        fitted = model.fit(X, y)
        assert fitted is model

        # Predict
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

        # Predict labels
        pred = model.predict(X)
        assert pred.shape == (X.shape[0],)
        assert set(np.unique(pred)).issubset({0, 1})


class TestModelFactory:
    def test_create_quantum_model(self):
        """Test creating quantum model via factory."""
        config = ModelConfig(type="vqc", n_qubits=4, max_iter=5, shots=100)
        model = ModelFactory.create_model(config, use_mock=True)
        assert isinstance(model, BaseEstimator)

    def test_create_classical_model(self):
        """Test creating classical model via factory."""
        config = ModelConfig(type="xgboost", n_estimators=10)
        model = ModelFactory.create_model(config, use_mock=True)
        assert isinstance(model, BaseEstimator)

    def test_create_model_convenience(self):
        """Test create_model convenience function."""
        config = {"type": "vqc", "n_qubits": 4, "max_iter": 5}
        model = create_model(config, use_mock=True)
        assert isinstance(model, BaseEstimator)

    def test_unknown_model_raises(self):
        """Test that unknown model type raises error."""
        config = ModelConfig(type="unknown_model")
        with pytest.raises(ValueError):
            ModelFactory.create_model(config)

    def test_list_models(self):
        """Test listing available models."""
        quantum = ModelFactory.list_quantum_models()
        classical = ModelFactory.list_classical_models()
        assert "vqc" in quantum
        assert "qsvm" in quantum
        assert "xgboost" in classical
        assert "logistic" in classical


class TestModelConfigCompatibility:
    def test_config_to_model_params(self):
        """Test that ModelConfig parameters map to model constructor."""
        config = ModelConfig(
            type="vqc",
            n_qubits=6,
            layers=2,
            ansatz="basic_entangling",
            encoding="amplitude",
            optimizer="sgd",
            learning_rate=0.05,
            max_iter=20,
            shots=512,
        )
        model = create_model(config, use_mock=True)
        params = model.get_params()
        assert params["n_qubits"] == 6
        assert params["layers"] == 2
        assert params["ansatz"] == "basic_entangling"
        assert params["encoding"] == "amplitude"
        assert params["optimizer"] == "sgd"
        assert params["learning_rate"] == 0.05
        assert params["max_iter"] == 20
        assert params["shots"] == 512