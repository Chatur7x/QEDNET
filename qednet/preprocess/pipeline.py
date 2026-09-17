"""Leakage-safe preprocessing pipeline (D3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .imputation import SafeImputer
from .scaling import SafeScaler
from .selection import SafeFeatureSelector
from .compression import SafePCA, AutoencoderCompressor


@dataclass
class PreprocessConfig:
    impute_strategy: str = "median"
    scale_method: str = "standard"
    select_method: str = "mutual_info"
    select_k: int = 16
    variance_threshold: float = 0.0
    compress_method: str = "pca"  # pca, autoencoder, none
    compress_dim: int = 8
    ae_hidden_dims: tuple = (64, 32)
    ae_epochs: int = 50
    smote: bool = False
    smote_k: int = 5
    random_state: int = 42


class LeakageSafePipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessing that prevents data leakage.
    All fit steps occur only on training data within CV folds.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._pipeline = None
        self._smote = None
        self._fitted = False

    def _build_pipeline(self, n_features: int):
        steps = [
            ("imputer", SafeImputer(strategy=self.config.impute_strategy)),
            ("scaler", SafeScaler(method=self.config.scale_method)),
        ]
        if self.config.select_method != "none" and self.config.select_k > 0:
            steps.append(("selector", SafeFeatureSelector(
                method=self.config.select_method,
                k=min(self.config.select_k, n_features),
                variance_threshold=self.config.variance_threshold,
                random_state=self.config.random_state,
            )))
        if self.config.compress_method == "pca":
            steps.append(("compressor", SafePCA(
                n_components=min(self.config.compress_dim, n_features),
                random_state=self.config.random_state,
            )))
        elif self.config.compress_method == "autoencoder":
            steps.append(("compressor", AutoencoderCompressor(
                input_dim=n_features,
                latent_dim=min(self.config.compress_dim, n_features),
                hidden_dims=self.config.ae_hidden_dims,
                epochs=self.config.ae_epochs,
                random_state=self.config.random_state,
            )))
        self._pipeline = Pipeline(steps)

        if self.config.smote:
            try:
                from imblearn.over_sampling import SMOTE
                self._smote = SMOTE(k_neighbors=self.config.smote_k, random_state=self.config.random_state)
            except ImportError:
                self._smote = None

    def fit(self, X, y=None):
        self._build_pipeline(X.shape[1])
        X_trans = self._pipeline.fit_transform(X, y)
        if self._smote is not None and y is not None:
            X_trans, y = self._smote.fit_resample(X_trans, y)
        self._fitted = True
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted")
        return self._pipeline.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if not self._fitted:
            return np.array([f"feat_{i}" for i in range(self.config.compress_dim)])
        return self._pipeline.get_feature_names_out(input_features)


def make_preprocess_pipeline(config: Optional[PreprocessConfig] = None) -> LeakageSafePipeline:
    """Factory for preprocessing pipeline."""
    return LeakageSafePipeline(config)


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """Reproducible train/test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )