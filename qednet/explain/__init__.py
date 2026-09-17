"""D9 Explainability public API."""

from .shap_explainer import SHAPExplainer, explain_instance
from .quantum_sensitivity import (
    quantum_input_sensitivity,
    parameter_sensitivity,
    finite_difference_sensitivity,
)
from .entanglement_explainer import entanglement_explainer, circuit_entanglement_profile
from .counterfactual import find_counterfactual, generate_counterfactuals
from .bottleneck_honesty import BottleneckHonestyMeter, bottleneck_honesty_check

__all__ = [
    "SHAPExplainer",
    "explain_instance",
    "quantum_input_sensitivity",
    "parameter_sensitivity",
    "finite_difference_sensitivity",
    "entanglement_explainer",
    "circuit_entanglement_profile",
    "find_counterfactual",
    "generate_counterfactuals",
    "BottleneckHonestyMeter",
    "bottleneck_honesty_check",
]