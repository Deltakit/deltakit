# (c) Copyright Riverlane 2020-2025.
"""Module which defines the abstract protocols for gates."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    ClassVar,
    Generator,
    Generic,
    Mapping,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import stim
from deltakit_circuit._qubit_identifiers import MeasurementRecord, Qubit, SweepBit, T, U


class PauliBasis(Enum):
    """The different Pauli bases for measurement and reset gates."""

    X = "X"
    Y = "Y"
    Z = "Z"


class Gate(ABC, Generic[T]):
    """Abstract gate class from which all other gate classes must inherit.

    Attributes
    ----------
    stim_string: str
        The string that stim associates to this gate.
    """

    stim_string: ClassVar[str]

    @property
    @abstractmethod
    def qubits(self) -> Tuple[Qubit[T], ...]:
        """Get all the qubits this gate acts on."""

    @abstractmethod
    def transform_qubits(self, id_mapping: Mapping[T, U]):
        """
        Transform this gates's qubits according to the id mapping.
        No transformation is performed if the qubits id is not in the mapping.

        Parameters
        ----------
        id_mapping : Mapping[T, U]
            A mapping of qubit types to other qubit types
        """

    @abstractmethod
    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget, ...]:
        """Convert the qubits this gate acts on to equivalent stim targets."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Define equality between gates."""

    @abstractmethod
    def __hash__(self) -> int:
        """Define unique hash for this gate."""

    @abstractmethod
    def __repr__(self) -> str:
        """Define the representation of this gate."""


class OneQubitGate(Gate[T]):
    """Abstraction of a single qubit gate. This is the base class for all one
    qubit gates including measurements, resets and Clifford gates.

    Attributes
    ----------
    stim_string: str
        The string that stim associates to this gate.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.
    """

    def __init__(self, qubit: Qubit[T] | T):
        self._qubit = Qubit(qubit) if not isinstance(qubit, Qubit) else qubit

    @property
    def qubit(self) -> Qubit[T]:
        """Get the single qubit that this gate acts on."""
        return self._qubit

    @property
    def qubits(self) -> Tuple[Qubit[T]]:
        return (self.qubit,)

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        if (new_id := id_mapping.get(self._qubit.unique_identifier)) is not None:
            self._qubit = Qubit(new_id)

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget, ...]:
        return (stim.GateTarget(qubit_mapping[self.qubit]),)

    def __repr__(self) -> str:
        return f"{self.stim_string}({self.qubit})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.qubit == other.qubit

    def __hash__(self) -> int:
        return hash((self.__class__, self.qubit))


class OneQubitCliffordGate(OneQubitGate[T]):
    """Gate for the abstract single qubit Clifford gate.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Attributes
    ----------
    stim_string: str
        The string that stim associates with this gate.
    """


class OneQubitResetGate(OneQubitGate[T]):
    """Gate for the abstract single qubit reset gate.
    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Attributes
    ----------
    stim_string: str
        The string that stim associates with this gate.
    """


class OneQubitMeasurementGate(OneQubitGate[T]):
    """Gate for the abstract single qubit measurement gate.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.

    Attributes
    ----------
    stim_string: str
        The string that stim associates to this gate.
    basis: PauliBasis
        The basis for the measurement.
    """

    basis: ClassVar[PauliBasis | None]

    def __init__(
        self, qubit: Qubit[T] | T, probability: float = 0.0, invert: bool = False
    ):
        super().__init__(qubit)
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between zero and one.")
        self._probability = probability
        self._is_inverted = invert

    @property
    def probability(self) -> float:
        """Get the probability of an error occurring when this gate is
        applied."""
        return self._probability

    @property
    def is_inverted(self) -> bool:
        """Get whether the measurement of this gate is inverted."""
        return self._is_inverted

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget, ...]:
        if self.is_inverted:
            return (stim.target_inv(qubit_mapping[self.qubit]),)
        return super().stim_targets(qubit_mapping)

    def approx_equals(
        self, other: object, *, rel_tol: float = 1e-9, abs_tol: float = 0
    ) -> bool:
        """Determine whether two measurement gates are approximately equal
        within a tolerance. The tolerance accounts for small differences in
        the error probabilities of the two classes. All other properties must
        be equal.

        Parameters
        ----------
        other : object
            The other object to compare this measurement gate to.
        rel_tol : float
            The allowed relative difference between the error probabilities of
            the two measurement gates, if this is larger than that calculated
            from abs_tol. Has the same meaning as in math.isclose.
            By default, 1e-9.
        abs_tol : float, optional
            The allowed absolute difference between the error probabilities
            of the two measurement gates, if this is larger than that
            calculate from rel_tol. Has the same meaning as in math.isclose.
            By default, 0.0.

        Returns
        -------
        bool
        """
        return (
            isinstance(other, self.__class__)
            and self.qubit == other.qubit
            and self.is_inverted == other.is_inverted
            and math.isclose(
                self.probability, other.probability, rel_tol=rel_tol, abs_tol=abs_tol
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.qubit == other.qubit
            and self.probability == other.probability
            and self.is_inverted == other.is_inverted
        )

    def __hash__(self) -> int:
        return hash((self.__class__, self.qubit, self.probability, self.is_inverted))

    def __invert__(self) -> OneQubitMeasurementGate[T]:
        """Invert the outcome of this measurement."""
        return self.__class__(
            self.qubit, self.probability, invert=not self._is_inverted
        )

    def __repr__(self) -> str:
        return (
            f"{'!' if self.is_inverted else ''}"
            f"{self.stim_string}({self.qubit}, probability={self.probability})"
        )


UT = TypeVar("UT", bound=Union[Qubit, SweepBit, MeasurementRecord])
VT = TypeVar("VT", bound=Union[Qubit, SweepBit, MeasurementRecord])

TwoOperandGateT = TypeVar("TwoOperandGateT", bound="TwoOperandGate")


class TwoOperandGate(Gate, Generic[UT, VT]):
    """Abstraction of a Clifford gate which take two generic operands.

    Parameters
    ----------
    operand1: UT | T
        The first operand of this operation. If the argument is not a Qubit,
        SweepBit or MeasurementRecord then the input is made into a Qubit.
    operand2: VT | T
        The second operand of this operation. If the argument is not a Qubit,
        SweepBit or MeasurementRecord then the input is made into a Qubit.

    Attributes
    ----------
    stim_string: str
        The string that stim associates with this gate.
    """

    stim_string: ClassVar[str]

    def __init__(self, operand1: UT | T, operand2: VT | T):
        operand1 = cast(
            UT,
            operand1
            if isinstance(operand1, (Qubit, SweepBit, MeasurementRecord))
            else Qubit(operand1),
        )
        operand2 = cast(
            VT,
            operand2
            if isinstance(operand2, (Qubit, SweepBit, MeasurementRecord))
            else Qubit(operand2),
        )
        if operand1 == operand2:
            raise ValueError("Operands for two qubit gates must be different.")
        self._operand1 = operand1
        self._operand2 = operand2

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        if (
            isinstance(self._operand1, Qubit)
            and (new_id1 := id_mapping.get(self._operand1.unique_identifier))
            is not None
        ):
            self._operand1 = Qubit(new_id1)  # type: ignore[assignment]
        if (
            isinstance(self._operand2, Qubit)
            and (new_id2 := id_mapping.get(self._operand2.unique_identifier))
            is not None
        ):
            self._operand2 = Qubit(new_id2)  # type: ignore[assignment]

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> Tuple[stim.GateTarget, stim.GateTarget]:
        """Get the stim gate targets which define this operation."""
        return (
            self._operand1.stim_targets(qubit_mapping)[0],
            self._operand2.stim_targets(qubit_mapping)[0],
        )

    @classmethod
    def from_consecutive(
        cls: Type[TwoOperandGateT], pairs: Sequence[UT | VT | T]
    ) -> Generator[TwoOperandGateT, None, None]:
        """Yield an class instance for each pair in a flattened sequence of
        data.

        Parameters
        ----------
        pairs : Sequence[UT | VT | T]
            The flat sequence of data. Length must be a multiple of 2.

        Yields
        ------
        Generator[TwoOperandOperationT, None, None]
            Qubit pair from neighbouring elements in the sequence.
        """
        if len(pairs) % 2 != 0:
            raise ValueError(
                "Two qubit gates can only be constructed from an even number of qubits"
            )
        for control, target in zip(pairs[::2], pairs[1::2], strict=True):
            yield cls(control, target)

    def __repr__(self) -> str:
        return f"{self.stim_string}({self._operand1}, {self._operand2})"


class SymmetricTwoQubitGate(TwoOperandGate[Qubit[T], Qubit[T]]):
    """Abstraction of a two-qubit gate which is symmetric in the control and
    target qubits. This means equality can be relaxed between two symmetric
    gates.

    Parameters
    ----------
    operand1: Qubit[T] | T
        The first operand of this operation. If the argument is not a Qubit
        then the input is made into a Qubit.
    operand2: Qubit[T] | T
        The second operand of this operation. If the argument is not a Qubit
        then the input is made into a Qubit.

    Attributes
    ----------
    stim_string: str
        The string that stim associates with this gate.
    """

    @property
    def qubits(self) -> Tuple[Qubit[T], Qubit[T]]:
        return (self._operand1, self._operand2)

    def transform_qubits(  # type: ignore[override]
        self, id_mapping: Mapping[T, U]
    ):
        if (new_id1 := id_mapping.get(self._operand1.unique_identifier)) is not None:
            self._operand1 = Qubit(new_id1)
        if (new_id2 := id_mapping.get(self._operand2.unique_identifier)) is not None:
            self._operand2 = Qubit(new_id2)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and set(self.qubits) == set(
            other.qubits
        )

    def __hash__(self) -> int:
        qubits = self.qubits
        return hash((self.__class__, frozenset(qubits)))


class ControlledGate(TwoOperandGate[UT, VT]):
    """Abstraction of a gate which can be represented as application of a
    one-qubit operation, conditional on the state of another qubit.

    Parameters
    ----------
    control: UT | T
        The first operand of this operation. If the argument is not a Qubit,
        SweepBit or MeasurementRecord then the input is made into a Qubit.
    target: VT | T
        The second operand of this operation. If the argument is not a Qubit,
        SweepBit or MeasurementRecord then the input is made into a Qubit.

    Attributes
    ----------
    stim_string: str
        The string that stim associates with this gate.
    """

    def __init__(self, control: UT | T, target: VT | T):
        # pylint: disable=useless-super-delegation
        super().__init__(control, target)

    @property
    def qubits(self) -> Tuple[Qubit[T], ...]:
        """Get all the qubits for this operation."""
        qubits = []
        if isinstance((operand1 := self._operand1), Qubit):
            qubits.append(operand1)
        if isinstance((operand2 := self._operand2), Qubit):
            qubits.append(operand2)
        return tuple(qubits)

    @property
    def control(self) -> UT:
        """Get the object which controls this gate."""
        return self._operand1

    @property
    def target(self) -> VT:
        """Get the target of this gate."""
        return self._operand2

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and (self.control, self.target) == (
            other.control,
            other.target,
        )

    def __hash__(self) -> int:
        return hash((self.__class__, self.control, self.target))

    def __repr__(self) -> str:
        return f"{self.stim_string}(control={self.control}, target={self.target})"
