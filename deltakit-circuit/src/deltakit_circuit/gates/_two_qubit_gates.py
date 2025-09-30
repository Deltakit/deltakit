# (c) Copyright Riverlane 2020-2025.
"""Module which provides all two-qubit gates.

In the gate class docstrings, the unitary matrices are defined so that the
first qubit state varies most rapidly, i.e., a vector representing the
two-qubit state has components (``|00>``, ``|10>``, ``|01>``, ``|11>``)^T,
where the first qubit is on the left and the second is on the right."""

from __future__ import annotations

from typing import ClassVar, Union, get_args

from deltakit_circuit.gates._abstract_gates import ControlledGate, SymmetricTwoQubitGate
from deltakit_circuit._qubit_identifiers import MeasurementRecord, Qubit, SweepBit, T

# pylint: disable=invalid-name


class CX(ControlledGate[Union[Qubit[T], SweepBit, MeasurementRecord], Qubit[T]]):
    """The Z-controlled X gate. First qubit is the control, second qubit is
    the target. The first qubit can be replaced by a measurement record.
    Applies an X gate to the target if the control is in the ``|1>`` state.

    Notes
    -----
    Negates the amplitude of the ``|1,->`` state.

    | Stabilizer Generators:
    |   ``X_ -> +XX``
    |   ``Z_ -> +Z_``
    |   ``_X -> +_X``
    |   ``_Z -> +ZZ``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 1 & 0 & 0
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "CX"


class CY(ControlledGate[Union[Qubit[T], SweepBit, MeasurementRecord], Qubit[T]]):
    """The Z-controlled Y gate. First qubit is the control, second qubit is
    the target. The first qubit can be replaced by a measurement record.
    Applies a Y gate to the target if the control is in the ``|1>`` state.

    Notes
    -----
    Negates the amplitude of the ``|1,-i>`` state.

    | Stabilizer Generators:
    |   ``X_ -> +XY``
    |   ``Z_ -> +Z_``
    |   ``_X -> +ZX``
    |   ``_Z -> +ZZ``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & -i \\\\
        0 & 0 & 1 & 0 \\\\
        0 & i & 0 & 0
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "CY"


class CZ(
    ControlledGate[Union[Qubit[T], SweepBit, MeasurementRecord], Qubit[T]],
    SymmetricTwoQubitGate,
):
    """The Z-controlled Z gate. First qubit is the control, second qubit is
    the target. Either qubit can be replaced by a measurement record. Applies
    a Z gate to the target if the control is in the ``|1>`` state.

    Notes
    -----
    Negates the amplitude of the ``|1,1>`` state.

    | Stabilizer Generators:
    |   ``X_ -> +XZ``
    |   ``Z_ -> +Z_``
    |   ``_X -> +ZX``
    |   ``_Z -> +_Z``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & -1
        \\end{pmatrix}
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and {self.control, self.target} == {
            other.control,
            other.target,
        }

    def __hash__(self) -> int:
        return hash((self.__class__, frozenset((self.control, self.target))))

    stim_string: ClassVar[str] = "CZ"


class SWAP(SymmetricTwoQubitGate[T]):
    """Swaps two qubits.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X_ -> +_X``
    |   ``Z_ -> +_Z``
    |   ``_X -> +X_``
    |   ``_Z -> +Z_``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SWAP"


class ISWAP(SymmetricTwoQubitGate[T]):
    """Swaps two qubits and phases the -1 eigenspace of the ZZ observable by i.
    Equivalent to `SWAP` then `CZ` then `S` on both targets.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X_ -> +ZY``
    |   ``Z_ -> +_Z``
    |   ``_X -> +YZ``
    |   ``_Z -> +Z_``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & i & 0 \\\\
        0 & i & 0 & 0 \\\\
        0 & 0 & 0 & 1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "ISWAP"


class ISWAP_DAG(SymmetricTwoQubitGate[T]):
    """Swaps two qubits and phases the -1 eigenspace of the ZZ observable by
    -i. Equivalent to `SWAP` then `CZ` then `S_DAG` on both targets.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X_ -> -ZY``
    |   ``Z_ -> +_Z``
    |   ``_X -> -YZ``
    |   ``_Z -> +Z_``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & -i & 0 \\\\
        0 & -i & 0 & 0 \\\\
        0 & 0 & 0 & 1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "ISWAP_DAG"


class SQRT_XX(SymmetricTwoQubitGate[T]):
    """Phases the -1 eigenspace of the XX observable by i.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X_ -> +X_``
    |   ``Z_ -> -YX``
    |   ``_X -> +_X``
    |   ``_Z -> -XY``

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1+i & 0 & 0 & 1-i \\\\
        0 & 1+i & 1-i & 0 \\\\
        0 & 1-i & 1+i & 0 \\\\
        1-i & 0 & 0 & 1+i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SQRT_XX"


class SQRT_XX_DAG(SymmetricTwoQubitGate[T]):
    """Phases the -1 eigenspace of the XX observable by -i.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X_ -> +X_``
    |   ``Z_ -> +YX``
    |   ``_X -> +_X``
    |   ``_Z -> +XY``

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1-i & 0 & 0 & 1+i \\\\
        0 & 1-i & 1+i & 0 \\\\
        0 & 1+i & 1-i & 0 \\\\
        1+i & 0 & 0 & 1-i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SQRT_XX_DAG"


class SQRT_YY(SymmetricTwoQubitGate[T]):
    """Phases the -1 eigenspace of the YY observable by i.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X_ -> -ZY``
    |   ``Z_ -> +XY``
    |   ``_X -> -YZ``
    |   ``_Z -> +YX``

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1+i & 0 & 0 & -1+i \\\\
        0 & 1+i & 1-i & 0 \\\\
        0 & 1-i & 1+i & 0 \\\\
        -1+i & 0 & 0 & 1+i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SQRT_YY"


class SQRT_YY_DAG(SymmetricTwoQubitGate[T]):
    """Phases the -1 eigenspace of the YY observable by -i.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X_ -> +ZY``
    |   ``Z_ -> -XY``
    |   ``_X -> +YZ``
    |   ``_Z -> -YX``

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1-i & 0 & 0 & -1-i \\\\
        0 & 1-i & 1+i & 0 \\\\
        0 & 1+i & 1-i & 0 \\\\
        -1-i & 0 & 0 & 1-i
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SQRT_YY_DAG"


class SQRT_ZZ(SymmetricTwoQubitGate[T]):
    """Phases the -1 eigenspace of the ZZ observable by i.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X_ -> +YZ``
    |   ``Z_ -> +Z_``
    |   ``_X -> +ZY``
    |   ``_Z -> +_Z``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & i & 0 & 0 \\\\
        0 & 0 & i & 0 \\\\
        0 & 0 & 0 & 1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SQRT_ZZ"


class SQRT_ZZ_DAG(SymmetricTwoQubitGate[T]):
    """Phases the -1 eigenspace of the ZZ observable by -i.

    Notes
    -----
    | Stabilizer Generators:
    |   ``X_ -> -YZ``
    |   ``Z_ -> +Z_``
    |   ``_X -> -ZY``
    |   ``_Z -> +_Z``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & -i & 0 & 0 \\\\
        0 & 0 & -i & 0 \\\\
        0 & 0 & 0 & 1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "SQRT_ZZ_DAG"


class XCX(
    ControlledGate[Qubit[T], Qubit[T]], SymmetricTwoQubitGate[Qubit[T]]
):
    """The X-controlled X gate. First qubit is the control, second qubit is
    the target. Applies an X gate to the target if the control is in the
    ``|->`` state.

    Notes
    -----
    Negates the amplitude of the ``|-,->`` state.

    | Stabilizer Generators:
    |   ``X_ -> +X_``
    |   ``Z_ -> +ZX``
    |   ``_X -> +_X``
    |   ``_Z -> +XZ``

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1 & 1 & 1 & -1 \\\\
        1 & 1 & -1 & 1 \\\\
        1 & -1 & 1 & 1 \\\\
        -1 & 1 & 1 & 1
        \\end{pmatrix}
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and {self.control, self.target} == {
            other.control,
            other.target,
        }

    def __hash__(self) -> int:
        return hash((self.__class__, frozenset((self.control, self.target))))

    stim_string: ClassVar[str] = "XCX"


class XCY(ControlledGate[Qubit[T], Qubit[T]]):
    """The X-controlled Y gate. First qubit is the control, second qubit is
    the target. Applies a Y gate to the target if the control is in the
    ``|->`` state.

    Notes
    -----
    Negates the amplitude of the ``|-,-i>`` state.

    | Stabilizer Generators:
    |   ``X_ -> +X_``
    |   ``Z_ -> +ZY``
    |   ``_X -> +XX``
    |   ``_Z -> +XZ``

    | Unitary Matrix:

    .. math::
        \\frac]{1}{2}
        \\begin{pmatrix}
        1 & 1 & -i & i \\\\
        1 & 1 & i & -i \\\\
        i & -i & 1 & 1 \\\\
        -i & i & 1 & 1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "XCY"


class XCZ(ControlledGate[Qubit[T], Union[Qubit[T], SweepBit, MeasurementRecord]]):
    """The X-controlled Z gate. First qubit is the control, second qubit is
    the target. The second qubit can be replaced by a measurement record.
    Applies a Z gate to the target if the control is in the ``|->`` state.

    Notes
    -----
    Negates the amplitude of the ``|-,1>`` state.

    | Stabilizer Generators:
    |   ``X_ -> +X_``
    |   ``Z_ -> +ZZ``
    |   ``_X -> +XX``
    |   ``_Z -> +_Z``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        0 & 0 & 1 & 0
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "XCZ"


class YCX(ControlledGate[Qubit[T], Qubit[T]]):
    """The Y-controlled X gate. First qubit is the control, second qubit is
    the target. Applies an X gate to the target if the control is in the
    ``|-i>`` state.

    Notes
    -----
    Negates the amplitude of the ``|-i,->`` state.

    | Stabilizer Generators:
    |   ``X_ -> +XX``
    |   ``Z_ -> +ZX``
    |   ``_X -> +_X``
    |   ``_Z -> +YZ``

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1 & -i & 1 & i \\\\
        i & 1 & -i & 1 \\\\
        1 & i & 1 & -i \\\\
        -i & 1 & i & 1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "YCX"


class YCY(
    ControlledGate[Qubit[T], Qubit[T]], SymmetricTwoQubitGate[Qubit[T]]
):
    """The Y-controlled Y gate. First qubit is the control, second qubit is
    the target. Applies a Y gate to the target if the control is in the
    ``|-i>`` state.

    Notes
    -----
    Negates the amplitude of the ``|-i,-i>`` state.

    | Stabilizer Generators:
    |   ``X_ -> +XY``
    |   ``Z_ -> +ZY``
    |   ``_X -> +YX``
    |   ``_Z -> +YZ``

    | Unitary Matrix:

    .. math::
        \\frac{1}{2}
        \\begin{pmatrix}
        1 & -i & -i & 1 \\\\
        i & 1 & -1 & -i \\\\
        i & -1 & 1 & -i \\\\
        1 & i & i & 1
        \\end{pmatrix}
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and {self.control, self.target} == {
            other.control,
            other.target,
        }

    def __hash__(self) -> int:
        return hash((self.__class__, frozenset((self.control, self.target))))

    stim_string: ClassVar[str] = "YCY"


class YCZ(ControlledGate[Qubit[T], Union[Qubit[T], SweepBit, MeasurementRecord]]):
    """The Y-controlled Z gate. First qubit is the control, second qubit is
    the target. The second qubit can be replaced by a measurement record.
    Applies a Z gate to the target if the control is in the ``|-i>`` state.

    Notes
    -----
    Negates the amplitude of the ``|-i,1>`` state.

    | Stabilizer Generators:
    |   ``X_ -> +XZ``
    |   ``Z_ -> +ZZ``
    |   ``_X -> +YX``
    |   ``_Z -> +_Z``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & -i \\\\
        0 & 0 & i & 0
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "YCZ"


class CXSWAP(ControlledGate[Qubit[T], Qubit[T]]):
    """A combination CX-and-SWAP gate. This gate is kak-equivalent
    to the ISWAP gate, but preserves X/Z noise bias. Equivalent to
    `CNOT` from target to control, immediately followed by another
    `CNOT` from control to target.

    Notes
    -----

    | Stabilizer Generators:
    |   ``X_ -> XX``
    |   ``Z_ -> _Z``
    |   ``_X -> X_``
    |   ``_Z -> ZZ``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        0 & 1 & 0 & 0
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "CXSWAP"


class CZSWAP(
    SymmetricTwoQubitGate, ControlledGate[Qubit[T], Qubit[T]]
):
    """A combination CZ-and-SWAP gate. This gate is kak-equivalent
    to the `ISWAP` gate. Equivalent to `H` on the target qubit, followed
    by `CNOT` from target to control, `CNOT` from control to target,
    and finally `H` on the control qubit.

    Notes
    -----

    | Stabilizer Generators:
    |   ``X_ -> ZX``
    |   ``Z_ -> _Z``
    |   ``_X -> XZ``
    |   ``_Z -> Z_``

    | Unitary Matrix:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & -1
        \\end{pmatrix}
    """

    stim_string: ClassVar[str] = "CZSWAP"


_TwoQubitGate = Union[
    CX,
    CXSWAP,
    CY,
    CZ,
    CZSWAP,
    ISWAP,
    ISWAP_DAG,
    SQRT_XX,
    SQRT_XX_DAG,
    SQRT_YY,
    SQRT_YY_DAG,
    SQRT_ZZ,
    SQRT_ZZ_DAG,
    SWAP,
    XCX,
    XCY,
    XCZ,
    YCX,
    YCY,
    YCZ,
]


TWO_QUBIT_GATES: frozenset[type[_TwoQubitGate]] = frozenset(get_args(_TwoQubitGate))
