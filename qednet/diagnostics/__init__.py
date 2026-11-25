"""Quantum diagnostics package (D7)."""

from .circuit_stats import circuit_stats, count_gates
from .entanglement import entanglement_entropy, measure_entanglement, meyer_wallach
from .gradients import barren_plateau_diagnostic, gradient_variance

__all__ = [
    "circuit_stats",
    "count_gates",
    "gradient_variance",
    "barren_plateau_diagnostic",
    "entanglement_entropy",
    "meyer_wallach",
    "measure_entanglement",
]
