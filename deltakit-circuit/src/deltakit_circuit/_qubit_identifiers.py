# (c) Copyright Riverlane 2020-2025.
"""An abstraction of instructions for stim."""

from __future__ import annotations

import warnings
from collections import abc
from itertools import chain
from typing import (
    ClassVar,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import stim

T = TypeVar("T", bound=Hashable)
U = TypeVar("U", bound=Hashable)


class Qubit(Generic[T]):
    """Abstraction of an immutable qubit type. This is generic over the
    provided type but that type must be hashable. This permits you to
    represent each qubit in whichever way is required. For most purposes,
    qubits will be identified using indices.

    Note if you want to use coordinates as the unique identifier you should
    use the Coordinate class in this module rather than a general tuple.

    Parameters
    ----------
    unique_identifier: T
        The unique way of identifying this qubit.
    stim_identifier: int | None
        Optional id for qubits in the context of a stim circuit, that being
        which int id they have in the stim circuit representation. Can be
        separate from the deltakit_circuit unique identifier.

    Examples
    --------
    >>> Qubit(0)
    Qubit(0)
    >>> Qubit("top left qubit")
    Qubit(top left qubit)
    >>> Qubit(Coordinate(0, 1, 2, 3))
    Qubit(Coordinate(0, 1, 2, 3))
    """

    def __init__(self, unique_identifier: T, stim_identifier: int | None = None):
        if isinstance(unique_identifier, abc.Generator):
            raise TypeError("Generators are not supported as qubit identifiers.")
        self._unique_identifier = unique_identifier

        self._stim_identifier: int | None
        if stim_identifier is None and isinstance(unique_identifier, int):
            self._stim_identifier = unique_identifier
        else:
            self._stim_identifier = stim_identifier

    @classmethod
    def pairs_from_consecutive(cls, ids: Sequence[T]) -> Iterator[Tuple[Qubit, Qubit]]:
        """A generator yielding pairs of qubit instances from a single
        sequence.

        Parameters
        ----------
        ids : Sequence[T]
            A sequence of IDs to convert into Qubit pairs. The pairs are
            assumed to sit next to each other.

        Yields
        ------
        Iterator[Tuple[Qubit, Qubit]]

        Raises
        ------
        ValueError
            If the length of the sequence of IDs is not even.
        """
        warnings.warn(
            "This method should not be used. Instead please use the "
            "`from_consecutive` method on the two qubit gate class",
            DeprecationWarning,
            stacklevel=2,
        )
        if len(ids) % 2 != 0:
            raise ValueError("Pairs cannot be constructed from an odd number of IDs.")
        for qubit1, qubit2 in zip(ids[::2], ids[1::2], strict=True):
            yield (cls(qubit1), cls(qubit2))

    @property
    def unique_identifier(self) -> T:
        """Get the object that uniquely identifies this qubit."""
        return self._unique_identifier

    @property
    def stim_identifier(self) -> int:
        """Get the integer that identifies this qubit in the context
        of a stim circuit.
        """
        if self._stim_identifier is None:
            raise ValueError(f"{self} has no stim identifier.")
        return self._stim_identifier

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget]:
        """Get the stim target gate target for this qubit."""
        return (stim.GateTarget(qubit_mapping[self]),)

    def permute_stim_circuit(
        self, stim_circuit: stim.Circuit, qubit_mapping: Mapping[Qubit[T], int]
    ):
        """Generate a stim circuit for a qubit. This is only a non empty
        circuit when T is a Coordinate."""
        if isinstance((coordinate := self.unique_identifier), Coordinate):
            # Get stim to construct the string since it will format (0,) into
            # (0, 0) so coordinates with single values are not output.
            stim_circuit.append("QUBIT_COORDS", qubit_mapping[self], tuple(coordinate))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Qubit)
            and self._unique_identifier == other._unique_identifier
        )

    def __hash__(self) -> int:
        return hash(self._unique_identifier)

    def __repr__(self) -> str:
        return f"Qubit({self._unique_identifier})"


class SweepBit:
    """Abstraction of a sweep bit as used by stim. A sweep bit represents a
    classical bit and is used as the control of a two-qubit gate.

    The index represents the index in a numpy array to read the classical bit
    from so the classical result doesn't need to be known at compile time, but
    only at sampling time. This makes using sweep bits different to using the
    results of measurements via `rec[-k]` because the sweep information is
    known separately from the circuit. The sweep bit array data must be stored
    in a separate file and passed in to stim when sampling."""

    def __init__(self, bit_index: int):
        if bit_index < 0:
            raise ValueError("Sweep bit index cannot be a negative number.")
        self._bit_index = bit_index

    @property
    def bit_index(self) -> int:
        """Get the bit index for this sweep bit."""
        return self._bit_index

    def stim_targets(self, *args) -> Tuple[stim.GateTarget]:
        # pylint: disable = unused-argument
        """Get this sweep bit as a stim gate target."""
        return (stim.GateTarget(stim.target_sweep_bit(self._bit_index)),)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SweepBit) and self._bit_index == other._bit_index

    def __hash__(self) -> int:
        return hash(self._bit_index)

    def __repr__(self) -> str:
        return f"SweepBit({self._bit_index})"


class Coordinate(Tuple[float, ...]):
    """Class which represents general coordinates.

    Parameters
    ----------
    *coordinates: float
        Any number of floats to specify the coordinate.
    """

    def __new__(cls, *coordinates: float):
        return super().__new__(cls, coordinates)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Coordinate) and super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = tuple.__hash__

    def __repr__(self) -> str:
        return f"Coordinate{super().__repr__()}"

    def __deepcopy__(self, memo):
        return self.__class__(*self)


class MeasurementRecord:
    """Reference to a measurement that has been made

    An example would be rec[-5] in stim.
    """

    def __init__(self, lookback_index: int):
        if lookback_index >= 0:
            raise ValueError("Lookback index should be negative.")
        self._lookback_index = lookback_index

    @property
    def lookback_index(self) -> int:
        """Get the lookback index"""
        return self._lookback_index

    def stim_targets(self, *args) -> Tuple[stim.GateTarget]:
        # pylint: disable = unused-argument
        """Get the stim target for this gate."""
        return (stim.GateTarget(stim.target_rec(self.lookback_index)),)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MeasurementRecord)
            and self._lookback_index == other._lookback_index
        )

    def __hash__(self) -> int:
        return hash(self._lookback_index)

    def __repr__(self) -> str:
        return f"MeasurementRecord({self.lookback_index})"


class PauliGate(Generic[T]):
    """Abstract representation of a Pauli gate which can be used in a Pauli
    Product class. This is required for the Measurement Pauli Product gate and
    also the Correlated Error noise models."""

    stim_identifier: ClassVar[str]

    def __init__(self, qubit: Qubit[T] | T):
        self._qubit = Qubit(qubit) if not isinstance(qubit, Qubit) else qubit

    @property
    def qubit(self) -> Qubit[T]:
        """Get the qubit that this gate acts on."""
        return self._qubit

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        """
        Transform this noise channel's qubits according to the id mapping.
        No transformation is performed if the qubits id is not in the mapping.

        Parameters
        ----------
        id_mapping : Mapping[T, U]
            A mapping of qubit types to other qubit types
        """
        if (new_id := id_mapping.get(self._qubit.unique_identifier)) is not None:
            self._qubit = Qubit(new_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InvertiblePauliGate):
            return other == self
        return (
            isinstance(other, PauliGate)
            and self.stim_identifier == other.stim_identifier
            and self.qubit == other.qubit
        )

    def __hash__(self) -> int:
        return hash((self.stim_identifier, self.qubit))

    def __repr__(self) -> str:
        return f"Pauli{self.stim_identifier}({self.qubit})"


class PauliX(PauliGate[T]):
    """Representation of an X gate on a single qubit which is used in Pauli
    products.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit to act the X gate on.
    """

    stim_identifier: ClassVar[str] = "X"

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget]:
        """Get the stim target for this gate."""
        return (stim.GateTarget(stim.target_x(qubit_mapping[self.qubit])),)


class PauliY(PauliGate[T]):
    """Representation of a Y gate on a single qubit which is used in Pauli
    products.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit to act the Y gate on.
    """

    stim_identifier: ClassVar[str] = "Y"

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget]:
        """Get the stim target for this gate."""
        return (stim.GateTarget(stim.target_y(qubit_mapping[self.qubit])),)


class PauliZ(PauliGate[T]):
    """Representation of a Z gate on a single qubit which is used in Pauli
    products.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit to act the Z gate on.
    """

    stim_identifier: ClassVar[str] = "Z"

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget]:
        """Get the stim target for this gate."""
        return (stim.GateTarget(stim.target_z(qubit_mapping[self.qubit])),)


class InvertiblePauliGate(PauliGate[T]):
    """Abstract representation of an invertible Pauli gate which is used in
    invertible Pauli products, such as the MPP gate.
    """

    def __init__(self, qubit: Qubit[T] | T, invert: bool = False):
        super().__init__(qubit)
        self._is_inverted = invert

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InvertiblePauliGate):
            return (
                self.qubit == other.qubit
                and self.stim_identifier == other.stim_identifier
                and self._is_inverted == other._is_inverted
            )
        return (
            isinstance(other, PauliGate)
            and self.qubit == other.qubit
            and self.stim_identifier == other.stim_identifier
            and not self._is_inverted
        )

    def __hash__(self) -> int:
        return super().__hash__() + int(self._is_inverted)

    def __repr__(self) -> str:
        return f"{'!' if self._is_inverted else ''}{super().__repr__()}"


class InvertiblePauliX(InvertiblePauliGate[T]):
    """Representation of an X gate measurement on a single qubit the result of
    which can be inverted. Needed for measuring Pauli products.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit to act the X gate on.
    invert: bool, optional
        Whether to invert the result of this gate measurement, by default
        False.
    """

    stim_identifier: ClassVar[str] = "X"

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget]:
        """Get the stim target for this gate."""
        return (
            stim.GateTarget(
                stim.target_x(qubit_mapping[self.qubit], invert=self._is_inverted)
            ),
        )

    def __invert__(self) -> InvertiblePauliX[T]:
        return InvertiblePauliX(self.qubit, invert=not self._is_inverted)


class InvertiblePauliY(InvertiblePauliGate[T]):
    """Representation of a Y gate measurement on a single qubit the result of
    which can be inverted. Needed for measuring Pauli products.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit to act the Y gate on.
    invert: bool, optional
        Whether to invert the result of this gate measurement, by default
        False.
    """

    stim_identifier: ClassVar[str] = "Y"

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget]:
        """Get the stim target for this gate."""
        return (
            stim.GateTarget(
                stim.target_y(qubit_mapping[self.qubit], invert=self._is_inverted)
            ),
        )

    def __invert__(self) -> InvertiblePauliY[T]:
        return InvertiblePauliY(self.qubit, invert=not self._is_inverted)


class InvertiblePauliZ(InvertiblePauliGate[T]):
    """Representation of a Z gate measurement on a single qubit the result of
    which can be inverted. Needed for measuring Pauli products.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit to act the Z gate on.
    invert: bool, optional
        Whether to invert the result of this gate measurement, by default
        False.
    """

    stim_identifier: ClassVar[str] = "Z"

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget]:
        """Get the stim target for this gate."""
        return (
            stim.GateTarget(
                stim.target_z(qubit_mapping[self.qubit], invert=self._is_inverted)
            ),
        )

    def __invert__(self) -> InvertiblePauliZ[T]:
        return InvertiblePauliZ(self.qubit, invert=not self._is_inverted)


_PauliGate = Union[PauliX[T], PauliY[T], PauliZ[T]]
_InvertiblePauliGate = Union[
    InvertiblePauliX[T], InvertiblePauliY[T], InvertiblePauliZ[T]
]


class MeasurementPauliProduct(Generic[T]):
    """A representation of a Pauli product which can include invertible Pauli
    gates. Required for the MPP instruction.

    Parameters
    ----------
    pauli_gates: Union[_PauliGate, _InvertiblePauliGate,
                       Iterable[Union[_PauliGate, _InvertiblePauliGate]]]
        Single Pauli or iterable of Pauli gates to use in this measurement.
        This must use the Pauli gates defined in this module and not those
        defined in the gates package.

    Examples
    --------
    >>> MeasurementPauliProduct(PauliX(Qubit(2)))
    [PauliX(Qubit(2))]
    >>> MeasurementPauliProduct(PauliX(i) for i in range(3))
    [PauliX(Qubit(0)), PauliX(Qubit(1)), PauliX(Qubit(2))]
    """

    def __init__(
        self,
        pauli_gates: Union[
            _PauliGate,
            _InvertiblePauliGate,
            Iterable[Union[_PauliGate, _InvertiblePauliGate]],
        ],
    ):
        pauli_gates = (
            (pauli_gates,)
            if isinstance(pauli_gates, (PauliGate, InvertiblePauliGate))
            else tuple(pauli_gates)
        )
        if len(pauli_gates) == 0:
            raise ValueError(
                "There must be at least one Pauli gate in a Pauli product."
            )
        qubits = [pauli_gate.qubit for pauli_gate in pauli_gates]
        if len(qubits) != len(set(qubits)):
            raise ValueError("Pauli product cannot contain duplicate qubits.")
        self._pauli_gates = pauli_gates

    @property
    def pauli_gates(self) -> Tuple[_PauliGate | _InvertiblePauliGate, ...]:
        """Get the gates for this Pauli product."""
        return self._pauli_gates

    @property
    def qubits(self) -> Tuple[Qubit[T], ...]:
        """Get all qubits for this gate in a tuple."""
        return tuple(gate.qubit for gate in self.pauli_gates)

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        """
        Transform this noise channel's qubits according to the id mapping.
        No transformation is performed if the qubits id is not in the mapping.

        Parameters
        ----------
        id_mapping : Mapping[T, U]
            A mapping of qubit types to other qubit types
        """
        for pauli_gate in self._pauli_gates:
            pauli_gate.transform_qubits(id_mapping)

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget, ...]:
        """Return all stim targets which specify this Pauli product."""
        # Create a list which is the same length as the final output and just
        # contains the stim combiners.
        with_combiners = [stim.target_combiner()] * (2 * len(self._pauli_gates) - 1)
        # Replace the even items of the combiner list with the appropriate
        # Pauli gate target.
        with_combiners[0::2] = chain.from_iterable(
            gate.stim_targets(qubit_mapping) for gate in self._pauli_gates
        )
        return tuple(with_combiners)

    def __iter__(self) -> Iterator[_PauliGate | _InvertiblePauliGate]:
        return iter(self._pauli_gates)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self._pauli_gates == other._pauli_gates
        )

    def __hash__(self) -> int:
        return hash(self._pauli_gates)

    def __repr__(self) -> str:
        return repr(list(self.pauli_gates))


class PauliProduct(MeasurementPauliProduct[T]):
    """A collection of Pauli gates which together make up a Pauli product.
    This is required for the correlated errors.

    Parameters
    ----------
    pauli_gates: _PauliGate | Iterable[_PauliGate]
        Single Pauli gate or iterable of Pauli gates which are in this product.
        This must use the Pauli gates defined in this module and not those
        defined in the gates package.

    Examples
    --------
    >>> PauliProduct(PauliX(Qubit(0)))
    [PauliX(Qubit(0))]
    >>> PauliProduct(PauliX(i) for i in range(3))
    [PauliX(Qubit(0)), PauliX(Qubit(1)), PauliX(Qubit(2))]
    """

    def __init__(self, pauli_gates: _PauliGate | Iterable[_PauliGate]):
        super().__init__(pauli_gates)
        self._pauli_gates: Tuple[_PauliGate]

    @property
    def pauli_gates(self) -> Tuple[_PauliGate, ...]:
        return self._pauli_gates

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget, ...]:
        """Return all stim targets which specify this Pauli product."""
        return tuple(
            chain.from_iterable(
                gate.stim_targets(qubit_mapping) for gate in self.pauli_gates
            )
        )

    def __iter__(self) -> Iterator[_PauliGate]:
        return iter(self._pauli_gates)
