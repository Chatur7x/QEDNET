"""Preprocessing pipeline with leakage prevention."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline that prevents data leakage.

    All transformers are fitted only on training data within CV folds.
    """

    def __init__(
        self,
        scaler: Literal["standard", "robust", "none"] = "standard",
        imputer: Literal["mean", "median", "none"] = "mean",
        compression: Literal["pca", "selectkbest", "none"] = "pca",
        n_components: int = 8,
        random_state: int = 42,
    ):
        self.scaler_type = scaler
        self.imputer_type = imputer
        self.compression_type = compression
        self.n_components = n_components
        self.random_state = random_state
        self._pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn pipeline."""
        steps = []

        # Imputer
        if self.imputer_type != "none":
            steps.append(("imputer", SimpleImputer(strategy=self.imputer_type)))

        # Scaler
        if self.scaler_type == "standard":
            steps.append(("scaler", StandardScaler()))
        elif self.scaler_type == "robust":
            steps.append(("scaler", RobustScaler()))

        # Compression
        if self.compression_type == "pca":
            steps.append(
                ("pca", PCA(n_components=self.n_components, random_state=self.random_state))
            )
        elif self.compression_type == "selectkbest":
            steps.append(("selectkbest", SelectKBest(f_classif, k=self.n_components)))

        return Pipeline(steps)

    def fit(self, X: NDArray, y: NDArray | None = None) -> "PreprocessingPipeline":
        """Fit the pipeline on training data only."""
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)
        return self

    def transform(self, X: NDArray) -> NDArray:
        """Transform data using fitted pipeline."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        return self._pipeline.transform(X)

    def fit_transform(self, X: NDArray, y: NDArray | None = None) -> NDArray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted.")
        if hasattr(self._pipeline, "get_feature_names_out"):
            names = list(self._pipeline.get_feature_names_out(input_features))
            return [f"component_{i}" for i in range(len(names))]
        n_features = self._pipeline.named_steps.get(
            "pca", self._pipeline.named_steps.get("selectkbest", None)
        )
        if n_features and hasattr(n_features, "n_components_"):
            return [f"component_{i}" for i in range(n_features.n_components_)]
        return input_features or []


def create_preprocessing_pipeline(config: dict) -> PreprocessingPipeline:
    """Create preprocessing pipeline from config dict."""
    return PreprocessingPipeline(
        scaler=config.get("scaler", "standard"),
        imputer=config.get("imputer", "mean"),
        compression=config.get("compression", "pca"),
        n_components=config.get("target_dim", 8),
        random_state=config.get("random_state", 42),
    )


# --- Leakage-safe assembler (merged, additive) ---
from dataclasses import dataclass  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402

from .compression import AutoencoderCompressor, SafePCA  # noqa: E402
from .imputation import SafeImputer  # noqa: E402
from .scaling import SafeScaler  # noqa: E402
from .selection import SafeFeatureSelector  # noqa: E402


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

    def __init__(self, config: PreprocessConfig | None = None):
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
            steps.append(
                (
                    "selector",
                    SafeFeatureSelector(
                        method=self.config.select_method,
                        k=min(self.config.select_k, n_features),
                        variance_threshold=self.config.variance_threshold,
                        random_state=self.config.random_state,
                    ),
                )
            )
        if self.config.compress_method == "pca":
            steps.append(
                (
                    "compressor",
                    SafePCA(
                        n_components=min(self.config.compress_dim, n_features),
                        random_state=self.config.random_state,
                    ),
                )
            )
        elif self.config.compress_method == "autoencoder":
            steps.append(
                (
                    "compressor",
                    AutoencoderCompressor(
                        input_dim=n_features,
                        latent_dim=min(self.config.compress_dim, n_features),
                        hidden_dims=self.config.ae_hidden_dims,
                        epochs=self.config.ae_epochs,
                        random_state=self.config.random_state,
                    ),
                )
            )
        self._pipeline = Pipeline(steps)

        if self.config.smote:
            try:
                from imblearn.over_sampling import SMOTE

                self._smote = SMOTE(
                    k_neighbors=self.config.smote_k, random_state=self.config.random_state
                )
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


def make_preprocess_pipeline(config: PreprocessConfig | None = None) -> LeakageSafePipeline:
    """Factory for preprocessing pipeline."""
    return LeakageSafePipeline(config)


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """Reproducible train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
