# (c) Copyright Riverlane 2020-2025.
"""Classes which represent noise resulting from qubits that leave the
preferred computational states."""

from __future__ import annotations

from typing import ClassVar

from deltakit_circuit._qubit_identifiers import T
from deltakit_circuit.noise_channels._abstract_noise_channels import (
    OneQubitOneProbabilityNoiseChannel,
)


class Leakage(OneQubitOneProbabilityNoiseChannel[T]):
    """The single qubit leakage channel.

    Whether or not this noise channel fires is recorded in a leakage table
    that tracks across all shots whether or not qubits are leaked at a
    given point in the circuit. When a qubit is leaked a 1 is recorded in
    the leakage table and the qubit is moved to the maximally mixed state by
    applying X_ERROR(0.5) and Z_ERROR(0.5). That is, a depolarising model
    of leakage is implemented through this channel.

    Leakage events are recorded and tracked so that the influence of leaked
    qubits on unleaked qubits across time and space can be accounted for.
    For example, if qubit 3 leaks and then a CX is performed on qubit 3 and 4,
    qubit 4 will also be fully depolarised.

    All leakage introduced by the leakage channel is undone by reset gates.
    That is, reset returns qubits to the computational subspace.

    This channel would typically be deployed alongside the RELAX channel
    which - in likeness of a reset gate - has the ability return a leaked
    qubit to the computational subspace.

    Parameters
    ----------
    qubit : Qubit[T] | T
        The qubit to apply leakage to.
    probability : float
        A single float specifying the leakage probability.
    Notes
    -----
    | Pauli Mixture:

        1-p: no change to the qubit's energy level, apply I
        p/4: record that the qubit has left the computational subspace, apply I
        p/4: record that the qubit has left the computational subspace, apply X
        p/4: record that the qubit has left the computational subspace, apply Y
        p/4: record that the qubit has left the computational subspace, apply Z
    """

    stim_string: ClassVar[str] = "LEAKAGE"


class Relax(OneQubitOneProbabilityNoiseChannel[T]):
    """The single qubit relaxation channel.

    Applies a single-qubit relaxation event with the given probability.
    Note: this is the probability that the target, when leaked, is returned
    to the computational subspace.

    If a qubit is relaxed, it is returned to the computational subspace
    and no longer marked as leaked. Once relaxed a qubit will, therefore,
    no longer depolarise qubits that it interacts with.

    Parameters
    ----------
    qubit : Qubit[T] | T
        The qubit to apply relaxation to.
    probability : float
        A single float specifying the relaxation probability.

    """

    stim_string: ClassVar[str] = "RELAX"
