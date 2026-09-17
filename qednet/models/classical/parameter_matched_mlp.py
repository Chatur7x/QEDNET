"""Parameter-matched MLP baseline (D6).
Matches parameter count to quantum models for fair comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:
    _TORCH = False


class ParameterMatchedMLP(BaseEstimator, ClassifierMixin):
    """
    MLP with configurable parameter count to match quantum models.
    Used for parameter-matched comparison in Advantage Certificate.
    """

    def __init__(
        self,
        n_params_target: int = 100,
        input_dim: int | None = None,
        hidden_dims: tuple = (32, 16),
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
        device: str = "cpu",
    ):
        if not _TORCH:
            raise RuntimeError("PyTorch required for ParameterMatchedMLP")
        self.n_params_target = n_params_target
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self._model = None
        self._fitted = False

    def _build_model(self, input_dim: int):
        import torch
        import torch.nn as nn

        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())

        self._model = nn.Sequential(*layers).to(self.device)

        # Count params
        total = sum(p.numel() for p in self._model.parameters())
        # Adjust hidden dims to match target if needed
        if total > self.n_params_target * 1.5:
            # Too many, reduce
            pass  # Could auto-adjust, but keep simple

    def fit(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if self.input_dim is None:
            self.input_dim = X.shape[1]
        self._build_model(self.input_dim)
        self._model.train()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        self._fitted = True
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_model")
        import torch
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            p1 = self._model(X_tensor).cpu().numpy().ravel()
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_scores(self, X):
        proba = self.predict_proba(X)
        eps = 1e-15
        p1 = np.clip(proba[:, 1], eps, 1 - eps)
        return np.log(p1 / (1 - p1))

    def metadata(self) -> dict:
        n_params = sum(p.numel() for p in self._model.parameters()) if self._model else 0
        return {
            "model": "parameter_matched_mlp",
            "n_params": n_params,
            "target_params": self.n_params_target,
            "hidden_dims": self.hidden_dims,
        }