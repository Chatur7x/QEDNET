"""Compression: PCA and Autoencoder (D3)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:
    _TORCH = False


class SafePCA(BaseEstimator, TransformerMixin):
    """Leakage-safe PCA wrapper."""

    def __init__(self, n_components: int = 8, random_state: int = 42, whiten: bool = False):
        self.n_components = n_components
        self.random_state = random_state
        self.whiten = whiten
        self._pca = None

    def fit(self, X, y=None):
        self._pca = PCA(n_components=self.n_components, random_state=self.random_state, whiten=self.whiten)
        self._pca.fit(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "_pca")
        return self._pca.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"PC{i+1}" for i in range(self.n_components)])

    @property
    def explained_variance_ratio_(self):
        check_is_fitted(self, "_pca")
        return self._pca.explained_variance_ratio_


class AutoencoderCompressor(BaseEstimator, TransformerMixin):
    """
    PyTorch-based autoencoder for nonlinear compression.
    Only fits on training data to prevent leakage.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: tuple = (64, 32),
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        random_state: int = 42,
        device: str = "cpu",
    ):
        if not _TORCH:
            raise RuntimeError("PyTorch required for AutoencoderCompressor")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self._model = None
        self._fitted = False

    def _build_model(self):
        layers = []
        prev = self.input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, self.latent_dim))
        self._encoder = nn.Sequential(*layers)

        layers = []
        prev = self.latent_dim
        for h in reversed(self.hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, self.input_dim))
        self._decoder = nn.Sequential(*layers)

        self._model = nn.Sequential(self._encoder, self._decoder).to(self.device)

    def fit(self, X, y=None):
        if not _TORCH:
            raise RuntimeError("PyTorch not available")
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._build_model()
        self._model.train()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        self._fitted = True
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("Autoencoder not fitted")
        import torch
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            encoded = self._encoder(X_tensor)
        return encoded.cpu().numpy()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"AE_{i+1}" for i in range(self.latent_dim)])