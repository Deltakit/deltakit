# (c) Copyright Riverlane 2020-2025.
"""Module which provides all one-qubit gates."""

from __future__ import annotations

from typing import ClassVar, get_args

from deltakit_circuit._qubit_identifiers import T
from deltakit_circuit.gates._abstract_gates import OneQubitCliffordGate

# pylint: disable=invalid-name


class I(OneQubitCliffordGate[T]):  # noqa: E742
    """Identity gate. Does nothing to the target qubit.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabiliser Generators:
    |   ``X -> +X``
    |   ``Z -> +Z``
    | Bloch Rotation:
    |   Axis:
    |   Angle: 0 degrees.

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & 1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "I"


class X(OneQubitCliffordGate[T]):
    """Pauli X gate. The bit flip gate.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> +X``
    |   ``Z -> -Z``
    | Bloch Rotation:
    |   Axis: +X,
    |   Angle: 180 degrees.

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        0 & 1 \\\\
        1 & 0
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "X"


class Y(OneQubitCliffordGate[T]):
    """Pauli Y gate.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> -X``
    |   ``Z -> -Z``
    | Bloch Rotation:
    |   Axis: +Y
    |   Angle: 180 degrees

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        0 & -i \\\\
        i & 0
        \\end{pmatrix}
    """

    stim_string = "Y"


class Z(OneQubitCliffordGate[T]):
    """Pauli Z gate. The phase flip gate.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> -X``
    |   ``Z -> +Z``
    | Bloch Rotation:
    |   Axis: +Z
    |   Angle: 180 degrees

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & -1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "Z"


class H(OneQubitCliffordGate[T]):
    """The Hadamard gate. Swaps the X and Z axes.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> +Z``
    |   ``Z -> +X``
    | Bloch Rotation:
    |   Axis: +X+Z
    |   Angle: 180 degrees

    | Unitary Matrix:

    .. math::
        \\frac{1}{\\sqrt{2}}
        \\begin{pmatrix}
        1 & 1 \\\\
        1 & -1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "H"


class H_XY(OneQubitCliffordGate[T]):
    """A variant of the Hadamard gate that swaps the X and Y axes (instead of
    X and Z).

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> +Y``
    |   ``Z -> -Z``
    | Bloch Rotation:
    |   Axis: +X+Y
    |   Angle: 180 degrees

    | Unitary Matrix:

    .. math::
        \\frac{1}{\\sqrt{2}}
        \\begin{pmatrix}
        0 & 1-i \\\\
        1+i & 0
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "H_XY"


class H_YZ(OneQubitCliffordGate[T]):
    """A variant of the Hadamard gate that swaps the Y and Z axes (instead of
    X and Z).

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> -X``
    |   ``Z -> +Y``
    | Bloch Rotation:
    |   Axis: +Y+Z
    |   Angle: 180 degrees

    | Unitary Matrix:

    .. math::
        \\frac{1}{\\sqrt{2}}
        \\begin{pmatrix}
        1 & -i \\\\
        +i & -1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "H_YZ"


class C_XYZ(OneQubitCliffordGate[T]):
    """Right handed period 3 axis cycling gate, sending X -> Y -> Z -> X.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> +Y``
    |   ``Z -> +X``
    | Bloch Rotation:
    |   Axis: +X+Y+Z
    |   Angle: 120 degrees

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1-i & -1-i \\\\
        1-i & 1+i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "C_XYZ"


class C_ZYX(OneQubitCliffordGate[T]):
    """Left handed period 3 axis cycling gate, sending Z -> Y -> X -> Z.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> +Z``
    |   ``Z -> +Y``
    | Bloch Rotation:
    |   Axis: +X+Y+Z
    |   Angle: -120 degrees

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1+i & 1+i \\\\
        -1+i & 1-i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "C_ZYX"


class S(OneQubitCliffordGate[T]):
    """Principal square root of Z gate. Phases the amplitude of ``|1>`` by i.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> +Y``
    |   ``Z -> +Z``
    | Bloch Rotation:
    |   Axis: +Z
    |   Angle: 90 degrees

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "S"


class S_DAG(OneQubitCliffordGate[T]):
    """Adjoint of the principal square root of Z gate. Phases the amplitude
    of ``|1>`` by -i.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> -Y``
    |   ``Z -> +Z``
    | Bloch Rotation:
    |   Axis: +Z
    |   Angle: -90 degrees

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & -i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "S_DAG"


class SQRT_X(OneQubitCliffordGate[T]):
    """Principal square root of X gate. Phases the amplitude of ``|->`` by i.
    Equivalent to `H` then `S` then `H`.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> +X``
    |   ``Z -> -Y``
    | Bloch Rotation:
    |   Axis: +X
    |   Angle: 90 degrees

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1+i & 1-i \\\\
        1-i & 1+i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SQRT_X"


class SQRT_X_DAG(OneQubitCliffordGate[T]):
    """Adjoint of the principal square root of X gate. Phases the amplitude
    of ``|->`` by -i. Equivalent to `H` then `S_DAG` then `H`.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> +X``
    |   ``Z -> +Y``
    | Bloch Rotation:
    |   Axis: +X
    |   Angle: -90 degrees

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1-i & 1+i \\\\
        1+i & 1-i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SQRT_X_DAG"


class SQRT_Y(OneQubitCliffordGate[T]):
    """Principal square root of Y gate. Phases the amplitude of ``|-i>`` by i.
    Equivalent to `S` then `H` then `S` then `H` then `S_DAG`.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> -Z``
    |   ``Z -> +X``
    | Bloch Rotation:
    |   Axis: +Y
    |   Angle: 90 degrees

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1+i & -1-i \\\\
        1+i & 1+i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SQRT_Y"


class SQRT_Y_DAG(OneQubitCliffordGate[T]):
    """Adjoint of the principal square root of Y gate. Phases the amplitude
    of ``|-i>`` by -i. Equivalent to `S` then `H` then `S_DAG` then `H` then
    `S_DAG`.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X -> +Z``
    |   ``Z -> -X``
    | Bloch Rotation:
    |   Axis: +Y
    |   Angle: -90 degrees

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1-i & 1-i \\\\
        -1+i & 1-i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SQRT_Y_DAG"


_OneQubitCliffordGate = (
    I
    | X
    | Y
    | Z
    | H
    | S
    | S_DAG
    | SQRT_X
    | SQRT_X_DAG
    | SQRT_Y
    | SQRT_Y_DAG
    | H_XY
    | H_YZ
    | C_XYZ
    | C_ZYX
)

ONE_QUBIT_GATES: frozenset[type[_OneQubitCliffordGate]] = frozenset(
    get_args(_OneQubitCliffordGate)
)
