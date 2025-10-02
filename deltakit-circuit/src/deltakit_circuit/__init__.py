# (c) Copyright Riverlane 2020-2025.
"""``deltakit.circuit`` provides classes to represent Stim circuit elements
and functions/methods for interacting with them."""

import importlib.metadata
from typing import TYPE_CHECKING

from deltakit_circuit._annotations._detector import Detector
from deltakit_circuit._annotations._observable import Observable
from deltakit_circuit._annotations._shift_coordinates import ShiftCoordinates
from deltakit_circuit._basic_types import Coord2D, Coord2DDelta
from deltakit_circuit._circuit import Circuit, Layer
from deltakit_circuit._detector_manipulation import trim_detectors
from deltakit_circuit._gate_layer import GateLayer
from deltakit_circuit._noise_factory import (
    GateReplacementPolicy,
    NoiseChannelGen,
    NoiseContext,
    NoiseProfile,
    after_clifford_depolarisation,
    after_reset_flip_probability,
    before_measure_flip_probability,
    measurement_noise_profile,
    noise_profile_with_inverted_noise,
)
from deltakit_circuit._noise_layer import NoiseLayer
from deltakit_circuit._qubit_identifiers import (
    Coordinate,
    InvertiblePauliX,
    InvertiblePauliY,
    InvertiblePauliZ,
    MeasurementPauliProduct,
    MeasurementRecord,
    PauliProduct,
    PauliX,
    PauliY,
    PauliZ,
    Qubit,
    SweepBit,
)

# Recursively import submodules for type checking, linting and code completion
if TYPE_CHECKING:  # pragma: no cover
    from deltakit_circuit import _annotations, gates, noise_channels

__version__ = importlib.metadata.version(__package__)

# Prevent import of non-public objects from this module.
del importlib, TYPE_CHECKING

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
