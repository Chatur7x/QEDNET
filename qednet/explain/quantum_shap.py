"""Quantum SHAP explainer for quantum models."""

from numpy.typing import NDArray

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class QuantumSHAP:
    """SHAP explainer adapted for quantum models.

    Uses gradient-free SHAP (KernelExplainer) since quantum circuits
    may not be differentiable in the classical sense.
    """

    def __init__(
        self,
        model,
        encoder,
        feature_names: list[str] | None = None,
        background_samples: int = 50,
    ):
        """Initialize Quantum SHAP explainer.

        Args:
            model: Fitted quantum model with predict_proba method
            encoder: Quantum encoder used for feature encoding
            feature_names: Optional feature names
            background_samples: Number of background samples for KernelExplainer
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap not installed. Run: pip install shap")
        self.model = model
        self.encoder = encoder
        self.feature_names = feature_names
        self.background_samples = background_samples
        self._explainer = None
        self._background = None

    def fit(self, X: NDArray, y: NDArray | None = None) -> "QuantumSHAP":
        """Fit the explainer."""
        # Use subset of data as background
        n_samples = min(self.background_samples, len(X))
        self._background = X[:n_samples]

        # Create wrapper that includes encoding
        def predict_wrapper(x):
            return self.model.predict_proba(x)[:, 1]

        self._explainer = shap.KernelExplainer(predict_wrapper, self._background)
        return self

    def explain(self, X: NDArray) -> NDArray:
        """Compute SHAP values for quantum model."""
        if self._explainer is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")
        return self._explainer.shap_values(X)

    def explain_single(self, x: NDArray) -> NDArray:
        """Explain single prediction."""
        return self.explain(x.reshape(1, -1))[0]

    def dependence_plot(self, feature_idx: int, X: NDArray) -> None:
        """Create SHAP dependence plot for a feature."""
        shap_values = self.explain(X)
        shap.dependence_plot(feature_idx, shap_values, X, feature_names=self.feature_names)
