"""Test configuration and fixtures."""

import numpy as np
import pytest
from numpy.typing import NDArray

from qednet.data.loader import Dataset
from qednet.models.base import BaseEstimator, BaseTransformer
from qednet.train.config import DatasetConfig, ExperimentConfig, ModelConfig


@pytest.fixture
def sample_data() -> tuple[NDArray, NDArray]:
    """Generate sample binary classification data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.int64)
    return X, y


@pytest.fixture
def sample_dataset() -> Dataset:
    """Create a sample dataset."""
    X, y = (
        np.random.randn(100, 10).astype(np.float32),
        np.random.randint(0, 2, 100).astype(np.int64),
    )
    return Dataset(
        X=X,
        y=y,
        feature_names=[f"feat_{i}" for i in range(10)],
        target_names=["class_0", "class_1"],
        name="test",
    )


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    """Create a test experiment configuration."""
    return ExperimentConfig(
        dataset=DatasetConfig(name="breast_cancer", target_dim=4),
        model=ModelConfig(type="vqc", n_qubits=4, layers=2, max_iter=10, shots=100),
    )


@pytest.fixture
def mock_estimator() -> BaseEstimator:
    """Create a mock estimator for testing."""

    class MockEstimator(BaseEstimator):
        def __init__(self):
            self.fitted = False

        def fit(self, X: NDArray, y: NDArray) -> "MockEstimator":
            self.classes_ = np.unique(y)
            self.fitted = True
            return self

        def predict_proba(self, X: NDArray) -> NDArray:
            if not self.fitted:
                raise RuntimeError("Not fitted")
            n_samples = X.shape[0]
            # Return random probabilities
            proba = np.random.rand(n_samples, 2)
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba

    return MockEstimator()


@pytest.fixture
def mock_transformer() -> BaseTransformer:
    """Create a mock transformer for testing."""

    class MockTransformer(BaseTransformer):
        def fit(self, X: NDArray, y: NDArray = None) -> "MockTransformer":
            self.fitted = True
            return self

        def transform(self, X: NDArray) -> NDArray:
            if not self.fitted:
                raise RuntimeError("Not fitted")
            return X

    return MockTransformer()
