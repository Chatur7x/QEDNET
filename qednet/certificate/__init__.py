"""Quantum Advantage Certificate components (D8 - quantum side).

A. Parameter-matched entanglement ablation
B. Dequantization support (RFF/Nystrom kernel outputs)
C. Quantum cost accounting
"""

from .ablation import (
    entanglement_ablation,
    EntanglementContribution,
    vqc_ablation_cost,
    qsvm_ablation_cost,
)
from .dequantization import (
    kernel_to_rff,
    kernel_to_nystrom,
    dequantize_kernel,
    DequantizationResult,
)
from .costing import quantum_cost, QuantumCost

__all__ = [
    "entanglement_ablation",
    "EntanglementContribution",
    "vqc_ablation_cost",
    "qsvm_ablation_cost",
    "kernel_to_rff",
    "kernel_to_nystrom",
    "dequantize_kernel",
    "DequantizationResult",
    "quantum_cost",
    "QuantumCost",
]