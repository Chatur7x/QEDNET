"""F4 — Quantum Encoding Library.

Reusable encoders with one consistent interface: ``encode(X)``.

Encoders:
  * AngleEncoder        — one RY(pi * x) per feature/qubit
  * ZZFeatureMap        — H + RZ(2x) + RZZ(2 x_i x_j) rings (Qiskit-style)
  * AmplitudeEncoder    — features as statevector amplitudes
  * ReuploadEncoder     — per-layer trainable-scaled feature re-injection

Every encoder validates input dimensions, qubit counts and feature sizes and
raises ``EncodingError`` for invalid inputs. Quantum encoding logic lives
ONLY here — models never duplicate it.
"""
from .base import BaseEncoder, EncodingError, EncodedBatch  # noqa: F401
from .angle import AngleEncoder  # noqa: F401
from .zz import ZZFeatureMap  # noqa: F401
from .amplitude import AmplitudeEncoder  # noqa: F401
from .reupload import ReuploadEncoder  # noqa: F401

ENCODERS = {
    "angle": AngleEncoder,
    "zz": ZZFeatureMap,
    "amplitude": AmplitudeEncoder,
    "reupload": ReuploadEncoder,
}


def get_encoder(name: str, **kwargs) -> BaseEncoder:
    if name not in ENCODERS:
        raise EncodingError(
            f"Unknown encoder '{name}'. Available: {sorted(ENCODERS)}")
    cls = ENCODERS[name]
    # accept only kwargs the encoder understands (reps applies to ZZ only)
    import inspect
    valid = set(inspect.signature(cls.__init__).parameters)
    usable = {k: v for k, v in kwargs.items() if k in valid}
    return cls(**usable)
