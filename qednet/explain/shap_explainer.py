"""SHAP explainability (D9)."""

from __future__ import annotations

import numpy as np
import warnings

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False


class SHAPExplainer:
    """
    Pipeline-level SHAP explainer that maps attributions back to original features.
    Handles the full pipeline: imputation -> scaling -> selection -> compression.
    """

    def __init__(self, model, preprocess_pipeline, feature_names: list[str]):
        if not _SHAP:
            raise RuntimeError("SHAP not installed")
        self.model = model
        self.preprocess = preprocess_pipeline
        self.feature_names = feature_names
        self._explainer = None
        self._background = None

    def fit_background(self, X_background: np.ndarray, n_samples: int = 100):
        """Fit SHAP background distribution on preprocessed data."""
        X_trans = self.preprocess.transform(X_background)
        # Subsample for speed
        if len(X_trans) > n_samples:
            idx = np.random.choice(len(X_trans), n_samples, replace=False)
            X_trans = X_trans[idx]
        self._background = X_trans
        self._explainer = shap.Explainer(self._predict_proba, X_trans)
        return self

    def _predict_proba(self, X_trans):
        """Wrapper for model prediction on transformed data."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_trans)[:, 1]
        elif hasattr(self.model, "decision_scores"):
            from scipy.special import expit
            return expit(self.model.decision_scores(X_trans))
        else:
            return self.model(X_trans).ravel()

    def explain(self, X: np.ndarray) -> dict:
        """Generate SHAP explanations for original feature space."""
        if self._explainer is None:
            self.fit_background(X)

        X_trans = self.preprocess.transform(X)
        shap_values = self._explainer(X_trans)

        # Map back to original features if possible
        # This is an approximation - true mapping requires knowing the compression
        original_shap = self._map_to_original(shap_values, X)

        return {
            "shap_values": shap_values.values,
            "base_values": shap_values.base_values,
            "data": shap_values.data,
            "feature_names": self._get_compressed_names(),
            "original_shap": original_shap,
        }

    def _map_to_original(self, shap_values, X_original):
        """Approximate mapping from compressed to original features."""
        # For PCA, we can use the components
        if hasattr(self.preprocess, "_pipeline"):
            for name, step in self.preprocess._pipeline.steps:
                if name == "compressor" and hasattr(step, "_pca"):
                    pca = step._pca
                    # Project SHAP values through PCA components
                    compressed_shap = shap_values.values
                    # SHAP on PCs -> SHAP on original features via components
                    original_shap = compressed_shap @ pca.components_
                    return original_shap
        # Fallback: uniform distribution
        return np.tile(shap_values.values.mean(axis=1, keepdims=True), (1, X_original.shape[1]))

    def _get_compressed_names(self):
        if hasattr(self.preprocess, "get_feature_names_out"):
            return self.preprocess.get_feature_names_out().tolist()
        return [f"feat_{i}" for i in range(self._background.shape[1])]


def explain_instance(
    model,
    preprocess_pipeline,
    X: np.ndarray,
    feature_names: list[str],
    instance_idx: int = 0,
) -> dict:
    """Explain a single prediction."""
    explainer = SHAPExplainer(model, preprocess_pipeline, feature_names)
    explainer.fit_background(X)
    result = explainer.explain(X[instance_idx:instance_idx+1])
    return {
        "instance_idx": instance_idx,
        "shap_values": result["shap_values"][0].tolist(),
        "base_value": float(result["base_values"][0]),
        "feature_names": result["feature_names"],
        "original_shap": result["original_shap"][0].tolist() if result["original_shap"] is not None else None,
    }