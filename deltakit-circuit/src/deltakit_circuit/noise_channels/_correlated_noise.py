# (c) Copyright Riverlane 2020-2025.
"""Module which gives abstractions for different correlated errors."""

from __future__ import annotations

from typing import ClassVar, get_args

from deltakit_circuit._qubit_identifiers import T
from deltakit_circuit.noise_channels._abstract_noise_channels import PauliProductNoise


class CorrelatedError(PauliProductNoise[T]):
    """Probabilistically applies a Pauli product error with a given
    probability.

    Sets the "correlated error occurred flag" to true if the
    error occurred. Otherwise sets the flag to false.

    Attributes:
        deltakit_stim_string: An identifier for the error type.
    """

    deltakit_stim_string: ClassVar[str] = "CORRELATED_ERROR"


class ElseCorrelatedError(PauliProductNoise[T]):
    """Probabilistically applies a Pauli product error with a given
    probability.

    The error is applied unless the "correlated error occurred flag" is set.
    If the error occurs, sets the "correlated error occurred flag" to true.
    Otherwise, leaves the flag alone.

    Attributes:
        deltakit_stim_string: An identifier for the error type.
    """

    deltakit_stim_string: ClassVar[str] = "ELSE_CORRELATED_ERROR"


_CorrelatedNoise = CorrelatedError[T] | ElseCorrelatedError[T]
ALL_CORRELATED_NOISE: frozenset[type[_CorrelatedNoise]] = frozenset(
    get_args(_CorrelatedNoise)
)
