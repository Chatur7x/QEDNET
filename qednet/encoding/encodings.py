"""D4 quantum encoding library (PennyLane, reusable by all models).

Encodings:
- angle: one RX rotation per feature (hardware friendly, O(1) depth).
- zz / iqp: Hadamard + RZ + RZZ entangling layer (standard QSVM map).
- amplitude: compact log2(d) qubits via QubitStateVector preparation.
- reupload: single-layer data re-injection primitive (full classifier in models).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .base import pad_to_pow2, validate_matrix, validate_vector

ENCODINGS = ("angle", "zz", "iqp", "amplitude", "reupload")


def _check_wires(n_qubits: int, wires) -> list:
    w = list(wires)
    if len(w) != n_qubits:
        raise ValueError(f"wires {len(w)} != n_qubits {n_qubits}")
    return w


def angle_encoding(x, wires, scale: float = 1.0) -> None:
    """RX(x_i * scale) on wire i. Requires n_features <= n_qubits."""
    n_qubits = len(list(wires))
    arr = validate_vector(x, n_qubits, "angle")
    w = _check_wires(n_qubits, wires)
    for i, v in enumerate(arr):
        qml.RX(float(v) * scale, wires=w[i])


def zz_feature_map(x, wires, reps: int = 1) -> None:
    """Hadamard + RZ(x) + chain RZZ(x_i*x_j); reps=1..3. Requires d <= n_qubits."""
    n_qubits = len(list(wires))
    arr = validate_vector(x, n_qubits, "zz")
    if reps < 1:
        raise ValueError("zz: reps must be >= 1")
    w = _check_wires(n_qubits, wires)
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=w[i])
        for i, v in enumerate(arr):
            qml.RZ(float(v), wires=w[i])
        for i in range(len(arr) - 1):
            qml.CNOT(wires=[w[i], w[i + 1]])
            qml.RZ(float(arr[i] * arr[i + 1]), wires=w[i + 1])
            qml.CNOT(wires=[w[i], w[i + 1]])


# Back-compat alias: spec names it ZZFeatureMap / IQP.
iqp_encoding = zz_feature_map


def amplitude_encoding(x, wires) -> None:
    """State preparation on log2(d) qubits; input is L2-normalized + padded."""
    n_qubits = len(list(wires))
    arr = validate_vector(x, n_qubits, "amplitude")
    w = _check_wires(n_qubits, wires)
    qml.StatePrep(pad_to_pow2(arr, n_qubits), wires=w)


def reupload_encoding(x, wires, theta) -> None:
    """One re-uploading layer: RY(x_i + theta_i) per wire. Primitive for D5."""
    n_qubits = len(list(wires))
    arr = validate_vector(x, n_qubits, "reupload")
    t = np.asarray(theta, dtype=float).ravel()
    if t.size != n_qubits:
        raise ValueError(f"reupload: theta size {t.size} != n_qubits {n_qubits}")
    w = _check_wires(n_qubits, wires)
    for i in range(n_qubits):
        v = float(arr[i]) if i < arr.size else 0.0
        qml.RY(v + float(t[i]), wires=w[i])


def get_encoding(name: str):
    key = name.lower()
    if key == "angle":
        return angle_encoding
    if key in ("zz", "iqp"):
        return zz_feature_map
    if key == "amplitude":
        return amplitude_encoding
    if key == "reupload":
        return reupload_encoding
    raise ValueError(f"unknown encoding {name!r}; choose from {ENCODINGS}")


def encode(name: str, x, wires, **kwargs) -> None:
    """Dispatcher so models share one call path (no duplicated logic)."""
    return get_encoding(name)(x, wires, **kwargs)


def statevector(x, n_qubits: int, encoding: str = "angle", **kwargs) -> np.ndarray:
    """Utility: return the encoded statevector (simulator, for tests/kernels)."""
    validate_vector(x, n_qubits, encoding)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def _circuit(v):
        encode(encoding, v, wires=range(n_qubits), **kwargs)
        return qml.state()

    return np.asarray(_circuit(np.asarray(x, dtype=float)))


def validate_dataset(X, n_qubits: int, encoding: str) -> np.ndarray:
    return validate_matrix(X, n_qubits, encoding)
