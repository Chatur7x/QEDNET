"""SHAP-based explainability."""

from typing: import Optional
import numpy as np
from numpy.typing import NDArray

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """SHAP explainer for model interpretability."""

    def __init__(self, model, feature_names: list[str] | None = None):
        """Initialize SHAP explainer.

        Args:
            model: Fitted model with predict_proba method
            feature_names: Optional feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap not installed. Run: pip install shap")
        self.model = model
        self.feature_names = feature_names
        self._explainer = None

    def fit(self, X: NDArray, y: NDArray | None = None) -> "SHAPExplainer":
        """Fit the explainer (compute expected values)."""
        # Use TreeExplainer for tree-based models, otherwise use KernelExplainer
        try:
            self._explainer = shap.TreeExplainer(self.model)
        except Exception:
            self._explainer = shap.KernelExplainer(self.model.predict_proba, X[:100])
        return self

    def explain(self, X: NDArray) -> NDArray:
        """Compute SHAP values."""
        if self._explainer is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")
        return self._explainer.shap_values(X)

    def explain_single(self, x: NDArray) -> NDArray:
        """Explain single prediction."""
        return self.explain(x.reshape(1, -1))[0]

    def summary_plot(self, X: NDArray, max_display: int = 20) -> None:
        """Create SHAP summary plot."""
        shap_values = self.explain(X)
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, max_display=max_display)