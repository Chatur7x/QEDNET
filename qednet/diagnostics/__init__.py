"""Quantum diagnostics package (D7)."""

from .circuit_stats import circuit_stats, count_gates
from .gradients import gradient_variance, barren_plateau_diagnostic
from .entanglement import entanglement_entropy, meyer_wallach, measure_entanglement

__all__ = [
    "circuit_stats",
    "count_gates",
    "gradient_variance",
    "barren_plateau_diagnostic",
    "entanglement_entropy",
    "meyer_wallach",
    "measure_entanglement",
]