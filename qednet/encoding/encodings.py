"""D4 quantum encoding library (PennyLane, reusable by all models)."""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .base import pad_to_pow2, validate_vector

ENCODINGS = ("angle", "zz", "amplitude", "reupload")


def angle_encoding(x, wires, scale: float = 1.0) -> None:
    """RX(x_i * scale) on wire i. Requires n_features <= n_qubits."""
    n_qubits = len(wires)
    arr = validate_vector(x, n_qubits, "angle")
    for i, v in enumerate(arr):
        qml.RX(float(v) * scale, wires=wires[i])


def zz_feature_map(x, wires, reps: int = 1) -> None:
    """Hadamard + RZ(x) + chain RZZ(x_i*x_j); reps=1..3. Requires d <= n_qubits."""
    n_qubits = len(wires)
    arr = validate_vector(x, n_qubits, "zz")
    if reps < 1:
        raise ValueError("zz: reps must be >= 1")
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=wires[i])
        for i, v in enumerate(arr):
            qml.RZ(float(v), wires=wires[i])
        for i in range(len(arr) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            qml.RZ(float(arr[i] * arr[i + 1]), wires=wires[i + 1])
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def amplitude_encoding(x, wires) -> None:
    """State preparation on log2(d) qubits; input is L2-normalized + padded."""
    n_qubits = len(wires)
    arr = validate_vector(x, n_qubits, "amplitude")
    qml.StatePrep(pad_to_pow2(arr, n_qubits), wires=wires)


def reupload_encoding(x, wires, theta) -> None:
    """One re-uploading layer: RY(x_i + theta_i) per wire. Primitive for D5."""
    n_qubits = len(wires)
    arr = validate_vector(x, n_qubits, "reupload")
    t = np.asarray(theta, dtype=float).ravel()
    if t.size != n_qubits:
        raise ValueError(f"reupload: theta size {t.size} != n_qubits {n_qubits}")
    for i in range(n_qubits):
        v = float(arr[i]) if i < arr.size else 0.0
        qml.RY(v + float(t[i]), wires=wires[i])


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
    return get_encoding(name)(x, wires, **kwargs)


def statevector(x, n_qubits: int, encoding: str = "angle", **kwargs) -> np.ndarray:
    validate_vector(x, n_qubits, encoding)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def _circuit(v):
        encode(encoding, v, wires=range(n_qubits), **kwargs)
        return qml.state()

    return np.asarray(_circuit(np.asarray(x, dtype=float)))
