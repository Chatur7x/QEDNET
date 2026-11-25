"""Quantum encoding strategies (D4 + Member C class-based API).

Member A (D4): Functional encoding API (angle_encoding, amplitude_encoding, etc.)
Member C: Class-based encoder API (AngleEncoder, AmplitudeEncoder, BasisEncoder) for pipeline integration
"""

# Member A's functional API (D4)
from .amplitude import AmplitudeEncoder
from .angle import AngleEncoder

# Member C's class-based API (for preprocessing/encoding pipeline)
from .base import BaseEncoder, pad_to_pow2, validate_vector
from .basis import BasisEncoder
from .encodings import (
    ENCODINGS,
    amplitude_encoding,
    angle_encoding,
    encode,
    get_encoding,
    reupload_encoding,
    statevector,
    zz_feature_map,
)

__all__ = [
    # Member A D4 functional API
    "ENCODINGS",
    "angle_encoding",
    "zz_feature_map",
    "amplitude_encoding",
    "reupload_encoding",
    "encode",
    "get_encoding",
    "statevector",
    "validate_vector",
    "pad_to_pow2",
    # Member C class-based API
    "BaseEncoder",
    "AngleEncoder",
    "AmplitudeEncoder",
    "BasisEncoder",
]
