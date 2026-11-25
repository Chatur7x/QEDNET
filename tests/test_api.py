"""Tests for FastAPI backend."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Mock the serving pipeline before importing
with patch("api.main.serving_pipeline", None), patch("api.main.model", None):
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

    def test_liveness_is_independent_of_model(self):
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readiness_requires_serving_pipeline(self):
        with patch("api.main.serving_pipeline", None):
            response = client.get("/ready")
        assert response.status_code == 503


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
                uncertainty=MagicMock(
                    variance=0.0001, std=0.01, ci_lower=0.73, ci_upper=0.77, shots=1024
                ),
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
        with patch("api.main.serving_pipeline", None):
            response = client.post("/predict", json={"features": [1.0]})  # Wrong number of features
            assert response.status_code == 503

    def test_predict_rejects_non_finite_features(self):
        from pydantic import ValidationError

        from api.main import PredictionRequest

        with pytest.raises(ValidationError, match="finite"):
            PredictionRequest(features=[float("nan")])

    def test_predict_passes_uncertainty_options_to_pipeline(self, mock_serving_pipeline):
        with patch("api.main.serving_pipeline", mock_serving_pipeline):
            response = client.post(
                "/predict",
                json={
                    "features": [1.0] * 30,
                    "return_uncertainty": False,
                    "return_conformal": False,
                },
            )
        assert response.status_code == 200
        mock_serving_pipeline.predict.assert_called_once()
        call_args = mock_serving_pipeline.predict.call_args
        np.testing.assert_array_equal(call_args.args[0], np.array([[1.0] * 30]))
        assert call_args.kwargs == {
            "return_uncertainty": False,
            "return_conformal": False,
        }


class TestBatchPredictEndpoint:
    def test_batch_predict(self):
        """Test batch prediction."""
        mock_pipeline = MagicMock()
        from api.main import PredictionResult

        mock_pipeline.predict.return_value = [
            PredictionResult(
                prediction=1,
                probability=0.7,
                calibrated_probability=0.7,
                routing="direct",
                uncertainty=None,
                conformal=None,
                model_name="test",
                config_hash="abc",
                abstained=False,
            )
            for _ in range(3)
        ]

        with patch("api.main.serving_pipeline", mock_pipeline):
            response = client.post(
                "/batch-predict", json={"features": [[1.0] * 30, [2.0] * 30, [3.0] * 30]}
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 3

    def test_batch_rejects_mismatched_feature_widths(self):
        response = client.post("/batch-predict", json={"features": [[1.0], [1.0, 2.0]]})
        assert response.status_code == 422


class TestExperimentEndpoint:
    def test_get_experiment(self):
        """Test get experiment endpoint."""
        response = client.get("/experiments/test_exp")
        assert response.status_code == 501
