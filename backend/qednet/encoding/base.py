"""Encoder base contract and shared utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import autograd.numpy as anp
import numpy as np

from ..sim.statevector import (BatchedCircuit, apply_cnot, apply_gate_2x2,
                               probs_pauliz, rz_matrix)


class EncodingError(ValueError):
    """Invalid encoding input (dims, qubits, feature size)."""


@dataclass
class EncodedBatch:
    """Consistent encoder output across all encoders.

    ``angles``: per-sample rotation angles (B, n_qubits) — consumed by
    variational circuits that embed data as rotations.
    ``states``: exact feature-map statevectors (B, 2**n) — consumed by
    kernel methods (QSVM). ``None`` when the map is parameter-dependent
    (data re-uploading).
    """
    angles: np.ndarray
    n_qubits: int
    states: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.angles is not None and self.angles.ndim != 2:
            raise EncodingError("angles must be 2-D (B, n_qubits)")


class BaseEncoder:
    """Interface every encoder implements."""

    name: str = "base"

    def __init__(self, n_qubits: Optional[int] = None):
        self._n_qubits = n_qubits

    # -- validation ---------------------------------------------------------
    def validate(self, X: np.ndarray, n_qubits: Optional[int] = None) -> None:
        X = np.asarray(X)
        if X.ndim != 2:
            raise EncodingError(f"{self.name}: X must be 2-D, got {X.ndim}-D")
        if X.shape[0] == 0:
            raise EncodingError(f"{self.name}: empty batch")
        if X.shape[1] == 0:
            raise EncodingError(f"{self.name}: zero features")
        if not np.all(np.isfinite(X)):
            raise EncodingError(
                f"{self.name}: non-finite values in X (impute/scale first)")
        nq = n_qubits if n_qubits is not None else self.n_qubits
        if nq is None or nq < 1 or nq > 12:
            raise EncodingError(
                f"{self.name}: invalid qubit count {nq} (must be 1..12)")

    @property
    def n_qubits(self) -> Optional[int]:
        return self._n_qubits

    def n_qubits_for(self, n_features: int) -> int:
        """Qubits required for a feature vector of ``n_features``."""
        raise NotImplementedError

    # -- main entry -----------------------------------------------------------
    def encode(self, X: np.ndarray) -> EncodedBatch:
        raise NotImplementedError

    # -- shared state helpers (exact, complex-safe) -----------------------------
    def _apply_rz(self, state, qubit: int, angles):
        return apply_gate_2x2(state, rz_matrix(angles), qubit)

    def _apply_cnot(self, state, control: int, target: int):
        return apply_cnot(state, control, target)

    def _zero_states(self, n_qubits: int, batch: int) -> np.ndarray:
        cir = BatchedCircuit(n_qubits)
        return cir.zero_state(batch, dtype=np.complex128)

    def _state_norms(self, states: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(np.abs(states) ** 2, axis=1))
