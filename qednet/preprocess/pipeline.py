"""Preprocessing pipeline with leakage prevention."""

from typing import Optional, Literal
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


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
        self._pipeline: Optional[Pipeline] = None

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
            steps.append(("pca", PCA(n_components=self.n_components, random_state=self.random_state)))
        elif self.compression_type == "selectkbest":
            steps.append(("selectkbest", SelectKBest(f_classif, k=self.n_components)))

        return Pipeline(steps)

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "PreprocessingPipeline":
        """Fit the pipeline on training data only."""
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)
        return self

    def transform(self, X: NDArray) -> NDArray:
        """Transform data using fitted pipeline."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        return self._pipeline.transform(X)

    def fit_transform(self, X: NDArray, y: Optional[NDArray] = None) -> NDArray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted.")
        if hasattr(self._pipeline, "get_feature_names_out"):
            return list(self._pipeline.get_feature_names_out(input_features))
        n_features = self._pipeline.named_steps.get("pca", self._pipeline.named_steps.get("selectkbest", None))
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