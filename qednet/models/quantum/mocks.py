"""Mock quantum models for development and testing."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from qednet.models.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier as SklearnMLP


class MockVQC(BaseEstimator):
    """Mock Variational Quantum Classifier using classical surrogate."""

    def __init__(
        self,
        n_qubits: int = 8,
        layers: int = 3,
        ansatz: str = "strongly_entangling",
        encoding: str = "angle",
        optimizer: str = "adam",
        learning_rate: float = 0.01,
        max_iter: int = 100,
        shots: int = 1024,
        random_state: int = 42,
        **kwargs,
    ):
        self.n_qubits = n_qubits
        self.layers = layers
        self.ansatz = ansatz
        self.encoding = encoding
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shots = shots
        self.random_state = random_state
        self._model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            C=1.0 / learning_rate,
        )

    def fit(self, X: NDArray, y: NDArray) -> "MockVQC":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        return self._model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "layers": self.layers,
            "ansatz": self.ansatz,
            "encoding": self.encoding,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "shots": self.shots,
            "random_state": self.random_state,
        }


class MockQSVM(BaseEstimator):
    """Mock Quantum SVM using classical SVM."""

    def __init__(
        self,
        n_qubits: int = 8,
        encoding: str = "angle",
        shots: int = 1024,
        random_state: int = 42,
        C: float = 1.0,
        **kwargs,
    ):
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.shots = shots
        self.random_state = random_state
        self.C = C
        self._model = SVC(probability=True, C=C, random_state=random_state)

    def fit(self, X: NDArray, y: NDArray) -> "MockQSVM":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        return self._model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "encoding": self.encoding,
            "shots": self.shots,
            "random_state": self.random_state,
            "C": self.C,
        }


class MockReuploadClassifier(BaseEstimator):
    """Mock Data Reuploading Classifier using MLP."""

    def __init__(
        self,
        n_qubits: int = 8,
        layers: int = 3,
        encoding: str = "angle",
        optimizer: str = "adam",
        learning_rate: float = 0.01,
        max_iter: int = 100,
        shots: int = 1024,
        random_state: int = 42,
    ):
        self.n_qubits = n_qubits
        self.layers = layers
        self.encoding = encoding
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shots = shots
        self.random_state = random_state
        self._model = SklearnMLP(
            hidden_layer_sizes=(n_qubits * 2,) * layers,
            max_iter=max_iter,
            learning_rate_init=learning_rate,
            random_state=random_state,
        )

    def fit(self, X: NDArray, y: NDArray) -> "MockReuploadClassifier":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        return self._model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "layers": self.layers,
            "encoding": self.encoding,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "shots": self.shots,
            "random_state": self.random_state,
        }


class MockHybridQNN(BaseEstimator):
    """Mock Hybrid Quantum-Classical Neural Network."""

    def __init__(
        self,
        n_qubits: int = 8,
        layers: int = 3,
        encoding: str = "angle",
        optimizer: str = "adam",
        learning_rate: float = 0.01,
        max_iter: int = 100,
        shots: int = 1024,
        random_state: int = 42,
    ):
        self.n_qubits = n_qubits
        self.layers = layers
        self.encoding = encoding
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shots = shots
        self.random_state = random_state
        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=layers * 2,
            random_state=random_state,
        )

    def fit(self, X: NDArray, y: NDArray) -> "MockHybridQNN":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        return self._model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "layers": self.layers,
            "encoding": self.encoding,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "shots": self.shots,
            "random_state": self.random_state,
        }


def get_mock_model(model_type: str, **kwargs) -> BaseEstimator:
    """Get mock model by type."""
    mocks = {
        "vqc": MockVQC,
        "qsvm": MockQSVM,
        "reupload": MockReuploadClassifier,
        "hybrid_qnn": MockHybridQNN,
    }
    if model_type not in mocks:
        raise ValueError(f"Unknown mock model type: {model_type}")
    return mocks[model_type](**kwargs)