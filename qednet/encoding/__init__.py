"""Quantum encoding strategies."""
from qednet.encoding.base import BaseEncoder
from qednet.encoding.angle import AngleEncoder
from qednet.encoding.amplitude import AmplitudeEncoder
from qednet.encoding.basis import BasisEncoder

__all__ = ["BaseEncoder", "AngleEncoder", "AmplitudeEncoder", "BasisEncoder"]