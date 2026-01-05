# (c) Copyright Riverlane 2020-2025.
"""This module provides all reset gates."""

from typing import ClassVar, get_args

from deltakit_circuit._qubit_identifiers import T
from deltakit_circuit.gates._abstract_gates import OneQubitResetGate, PauliBasis


class RZ(OneQubitResetGate[T]):
    r"""Z - basis reset. Forces each target qubit into the :math:`\ket{0}` state
    by silently measuring it in the Z basis and applying an X gate if it ended
    up in the :math:`\ket{1}` state.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   1 -> +Z

    | Decomposition(into H, S, CX, M, R):
    | The following circuit is equivalent(up to global phase) to ``R 0``
    |   R 0
    | (The decomposition is trivial because this gate is in the target gate
    | set.)
    """

    basis: ClassVar[PauliBasis] = PauliBasis.Z
    stim_string: ClassVar[str] = f"R{basis.value}"


class RX(OneQubitResetGate[T]):
    """X - basis reset. Forces each target qubit into the ``| + >`` state by
    silently measuring it in the X basis and applying a Z gate if it ended
    up in the ``| ->`` state.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   1 -> +X

    | Decomposition(into H, S, CX, M, R):
    | The following circuit is equivalent(up to global phase) to ``RX 0``
    |   H 0
    |   R 0
    |   H 0
    """

    basis: ClassVar[PauliBasis] = PauliBasis.X
    stim_string: ClassVar[str] = f"R{basis.value}"


class RY(OneQubitResetGate[T]):
    """Y - basis reset. Forces each target qubit into the ``| i >`` state by
    silently measuring it in the Y basis and applying an X gate if it ended
    up in the ``| -i >`` state.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   1 -> +Y

    | Decomposition(into H, S, CX, M, R):
    |   The following circuit is equivalent(up to global phase) to ``RY 0``
    |   S 0
    |   S 0
    |   S 0
    |   H 0
    |   R 0
    |   H 0
    |   S 0
    """

    basis: ClassVar[PauliBasis] = PauliBasis.Y
    stim_string: ClassVar[str] = f"R{basis.value}"


_ResetGate = RZ | RX | RY

RESET_GATES: frozenset[type[_ResetGate]] = frozenset(get_args(_ResetGate))
