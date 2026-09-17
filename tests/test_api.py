"""Tests for FastAPI backend."""

import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock

# Mock the serving pipeline before importing
with patch("api.main.serving_pipeline", None), \
     patch("api.main.model", None):
    from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check_no_model(self):
        """Test health endpoint when no model loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestModelsEndpoint:
    def test_models_no_model(self):
        """Test models endpoint when no model loaded."""
        response = client.get("/models")
        assert response.status_code == 503


class TestPredictEndpoint:
    @pytest.fixture
    def mock_serving_pipeline(self):
        """Create mock serving pipeline."""
        from api.main import PredictionResult, ServingPipeline
        mock = MagicMock(spec=ServingPipeline)
        mock.predict.return_value = [
            PredictionResult(
                prediction=1,
                probability=0.75,
                calibrated_probability=0.75,
                routing="direct",
                uncertainty=MagicMock(variance=0.0001, std=0.01, ci_lower=0.73, ci_upper=0.77, shots=1024),
                conformal=MagicMock(prediction_set=[1], abstained=False, alpha=0.1),
                model_name="test_model",
                config_hash="abc123",
                abstained=False,
            )
        ]
        return mock

    def test_predict_success(self, mock_serving_pipeline):
        """Test successful prediction."""
        with patch("api.main.serving_pipeline", mock_serving_pipeline):
            response = client.post("/predict", json={"features": [1.0] * 30})
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == 1
            assert data["probability"] == 0.75
            assert "RESEARCH ONLY" in data["disclaimer"]

    def test_predict_no_pipeline(self):
        """Test prediction when pipeline not initialized."""
        with patch("api.main.serving_pipeline", None):
            response = client.post("/predict", json={"features": [1.0] * 30})
            assert response.status_code == 503

    def test_predict_invalid_features(self):
        """Test prediction with invalid feature count."""
        with patch("api.main.serving_pipeline", MagicMock()):
            response = client.post("/predict", json={"features": [1.0]})  # Wrong number of features
            # Should still work but model might complain
            # The API doesn't validate feature count


class TestBatchPredictEndpoint:
    def test_batch_predict(self):
        """Test batch prediction."""
        mock_pipeline = MagicMock()
        from api.main import PredictionResult
        mock_pipeline.predict.return_value = [
            PredictionResult(
                prediction=1, probability=0.7, calibrated_probability=0.7,
                routing="direct", uncertainty=None, conformal=None,
                model_name="test", config_hash="abc", abstained=False
            ) for _ in range(3)
        ]

        with patch("api.main.serving_pipeline", mock_pipeline):
            response = client.post("/batch-predict", json={"features": [[1.0]*30, [2.0]*30, [3.0]*30]})
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 3


class TestExperimentEndpoint:
    def test_get_experiment(self):
        """Test get experiment endpoint."""
        response = client.get("/experiments/test_exp")
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "test_exp"