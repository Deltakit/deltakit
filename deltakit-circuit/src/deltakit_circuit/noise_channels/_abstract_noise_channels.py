# (c) Copyright Riverlane 2020-2025.
"""Module which defines the abstract noise channel protocols."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from typing import ClassVar, Generic, TypeVar

import stim
from typing_extensions import Self

from deltakit_circuit._qubit_identifiers import PauliProduct, Qubit, T, U, _PauliGate
from deltakit_circuit._stim_identifiers import NoiseStimIdentifier


class ProbabilityError(ValueError):
    """Error class which is raised if probability is outside of bounds."""

    def __init__(self):
        super().__init__("Probability must be between zero and one inclusive.")


class NoiseChannel(ABC, Generic[T]):
    """Abstract base noise channel which all other noise channel classes must
    implement.

    Attributes
    ----------
    stim_string: str
        The string that stim associates to this gate.
    """

    stim_string: ClassVar[str]

    def __init__(self, *_args, tag: str | None = None, **_kwargs) -> None:
        super().__init__()
        self._tag = tag

    @property
    def tag(self) -> str | None:
        return self._tag

    @property
    @abstractmethod
    def qubits(self) -> tuple[Qubit[T], ...]:
        """Get all the qubits for this noise channel."""

    @property
    @abstractmethod
    def stim_identifier(self) -> NoiseStimIdentifier:
        """Get the collection of things which uniquely define this object in
        the context of stim. Each noise channel is unique from its stim
        identifier and its probability.
        """

    @property
    @abstractmethod
    def probabilities(self) -> tuple[float, ...]:
        """Get all the probabilities for this noise channel"""

    @abstractmethod
    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> tuple[stim.GateTarget, ...]:
        """Return all the stim targets which specifies this noise channel."""

    @abstractmethod
    def transform_qubits(self, id_mapping: Mapping[T, U]):
        """
        Transform this noise channel's qubits according to the id mapping.
        No transformation is performed if the qubits id is not in the mapping.

        Parameters
        ----------
        id_mapping : Mapping[T, U]
            A mapping of qubit types to other qubit types
        """

    def approx_equals(
        self, other: object, *, rel_tol: float = 1e-9, abs_tol: float = 0
    ) -> bool:
        """Determine whether two noise channels are equal within a given
        tolerance. The tolerance accounts for differences in the probabilities
        and all other properties of the noise channel must be the same.

        Parameters
        ----------
        other : object
            The other object to compare this noise channel to.
        rel_tol : float
            The allowed relative difference between the error probabilities of
            the two noise channels, if this is larger than that calculated
            from abs_tol. Has the same meaning as in math.isclose.
            By default, 1e-9.
        abs_tol : float, optional
            The allowed absolute difference between the error probabilities
            of the two noise channels, if this is larger than that calculated
            from rel_tol. Has the same meaning as in math.isclose.
            By default, 0.0.

        Returns
        -------
        bool
        """
        return (
            isinstance(other, self.__class__)
            and self.qubits == other.qubits
            and len(self.probabilities) == len(other.probabilities)
            and all(
                math.isclose(self_prob, other_prob, rel_tol=rel_tol, abs_tol=abs_tol)
                for self_prob, other_prob in zip(
                    self.probabilities, other.probabilities, strict=True
                )
            )
        )

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Define equality between two different noise models."""

    @abstractmethod
    def __hash__(self) -> int:
        """Compute a hash of the noise model."""

    @abstractmethod
    def __repr__(self) -> str:
        """Define the representation of this noise channel."""


class OneProbabilityNoiseChannel(NoiseChannel[T]):
    """Abstract noise channel which contains only one probability.

    Attributes
    ----------
    stim_string: str
        The string that stim associates to this gate.

    Parameters
    ----------
    probability: float
        The probability that this error occurs.
    """

    def __init__(
        self, probability: float, *args, tag: str | None = None, **kwargs
    ) -> None:
        super().__init__(*args, tag=tag, **kwargs)
        if not 0 <= probability <= 1:
            raise ProbabilityError()
        self._probability = probability

    @property
    def probability(self) -> float:
        """Get the probability for this noise channel to be applied."""
        return self._probability

    @property
    def probabilities(self) -> tuple[float]:
        return (self._probability,)

    @property
    def stim_identifier(self) -> NoiseStimIdentifier:
        return NoiseStimIdentifier(
            self.__class__.stim_string, (self.probability,), self.tag
        )


class MultiProbabilityNoiseChannel(NoiseChannel[T]):
    """Abstract noise channel which contains multiple probabilities.

    Attributes
    ----------
    stim_string: str
        The string that stim associates to this gate.
    """

    def __init__(
        self, probabilities: tuple[float, ...], *args, tag: str | None = None, **kwargs
    ):
        super().__init__(*args, tag=tag, **kwargs)
        if not all(0 <= probability <= 1 for probability in probabilities):
            raise ProbabilityError()
        if sum(probabilities) > 1:
            msg = "The sum of probabilities cannot be greater than one."
            raise ValueError(msg)
        self._probabilities = probabilities

    @property
    def probabilities(self) -> tuple[float, ...]:
        """Get all the probabilities which define this noise channel."""
        return self._probabilities

    @property
    def stim_identifier(self) -> NoiseStimIdentifier:
        return NoiseStimIdentifier(
            self.__class__.stim_string, self.probabilities, self.tag
        )


OneQubitNoiseChannelT = TypeVar("OneQubitNoiseChannelT", bound="OneQubitNoiseChannel")


class OneQubitNoiseChannel(NoiseChannel[T]):
    """Abstract noise channel which only acts on a single qubit.

    Attributes
    ----------
    stim_string: str
        The string that stim associates to this gate.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this noise channel error acts on.
    """

    def __init__(self, qubit: Qubit[T] | T, *args, tag: str | None = None, **kwargs):
        super().__init__(*args, tag=tag, **kwargs)
        self._qubit = Qubit(qubit) if not isinstance(qubit, Qubit) else qubit

    @property
    def qubit(self) -> Qubit[T]:
        """Get the qubit that this noise channel acts on."""
        return self._qubit

    @property
    def qubits(self) -> tuple[Qubit[T]]:
        return (self.qubit,)

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        if (new_id := id_mapping.get(self._qubit.unique_identifier)) is not None:
            self._qubit = Qubit(new_id)

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> tuple[stim.GateTarget]:
        return (qubit_mapping[self._qubit],)

    @classmethod
    def generator_from_prob(
        cls, *gate_args, tag: str | None = None, **gate_kwargs
    ) -> Callable[[Iterable[Qubit[T]] | T], list[Self]]:
        """Return a classmethod that can be used to create a noise channel
        with a predetermined probability"""

        def inner_gen(qubits) -> list[Self]:
            return [cls(qubit, *gate_args, tag=tag, **gate_kwargs) for qubit in qubits]

        return inner_gen


TwoQubitNoiseChannelT = TypeVar("TwoQubitNoiseChannelT", bound="TwoQubitNoiseChannel")


class TwoQubitNoiseChannel(NoiseChannel[T]):
    """Abstract noise channel which acts on pairs of qubits.

    Attributes
    ----------
    stim_string: str
        The string that stim associates to this gate.

    Parameters
    ----------
    qubit1: Qubit[T] | T
        The first qubit in the noise channel.
    qubit2: Qubit[T] | T
        The second qubit in the noise channel.
    """

    def __init__(
        self,
        qubit1: Qubit[T] | T,
        qubit2: Qubit[T] | T,
        *args,
        tag: str | None = None,
        **kwargs,
    ):
        super().__init__(tag, *args, tag=tag, **kwargs)
        qubit1 = Qubit(qubit1) if not isinstance(qubit1, Qubit) else qubit1
        qubit2 = Qubit(qubit2) if not isinstance(qubit2, Qubit) else qubit2
        if qubit1 == qubit2:
            msg = "Qubits in two qubit noise channels must be different."
            raise ValueError(msg)
        self._qubit1, self._qubit2 = qubit1, qubit2

    @property
    def qubits(self) -> tuple[Qubit[T], Qubit[T]]:
        """Get all the qubits this noise channel acts on."""
        return (self._qubit1, self._qubit2)

    @property
    def qubit1(self) -> Qubit[T]:
        """Get the first qubit for this noise channel."""
        return self._qubit1

    @property
    def qubit2(self) -> Qubit[T]:
        """Get the second qubit for this noise channel."""
        return self._qubit2

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        if (new_id1 := id_mapping.get(self._qubit1.unique_identifier)) is not None:
            self._qubit1 = Qubit(new_id1)
        if (new_id2 := id_mapping.get(self._qubit2.unique_identifier)) is not None:
            self._qubit2 = Qubit(new_id2)

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> tuple[stim.GateTarget, stim.GateTarget]:
        """Get all stim gate targets for this noise channel in a tuple."""
        return (
            self._qubit1.stim_targets(qubit_mapping)[0],
            self._qubit2.stim_targets(qubit_mapping)[0],
        )

    @classmethod
    def generator_from_prob(
        cls, *gate_args, tag: str | None = None, **gate_kwargs
    ) -> Callable[[Sequence[Qubit[T] | T]], list[Self]]:
        """Return a classmethod that can be used to create a noise channel
        with a predetermined probability"""

        def inner_gen(qubits: Sequence[Qubit[T] | T]) -> list[Self]:
            return list(
                cls.from_consecutive(qubits, *gate_args, tag=tag, **gate_kwargs)
            )

        return inner_gen

    @classmethod
    def from_consecutive(
        cls,
        pairs: Sequence[Qubit[T] | T],
        *gate_args,
        tag: str | None = None,
        **gate_kwargs,
    ) -> Generator[Self, None, None]:
        """Yield a class instance for each pair in a flattened sequence of
        data.

        Parameters
        ----------
        pairs : Sequence[Qubit[T] | T]
            The flat sequence of data. Length must be a multiple of 2.
        tag : str | None
            An optional tag for the instruction.

        Yields
        ------
        Generator[Depolarise2, None, None]
            Qubit pair from neighbouring elements in the sequence.
        """
        if len(pairs) % 2 != 0:
            msg = (
                "Two qubit noise channels can only be "
                "constructed from an even number of qubits"
            )
            raise ValueError(msg)
        for qubit1, qubit2 in zip(pairs[::2], pairs[1::2], strict=True):
            yield cls(qubit1, qubit2, *gate_args, tag=tag, **gate_kwargs)


NC = TypeVar("NC", bound="OneQubitOneProbabilityNoiseChannel")


class OneQubitOneProbabilityNoiseChannel(
    OneQubitNoiseChannel[T], OneProbabilityNoiseChannel[T]
):
    """Abstract noise channel for noise channels which take a single qubit and
    a single probability such as Pauli X, Y or Z and Depolarise 1.

    Attributes
    ----------
    stim_string: str
        The string that stim associates to this gate.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that the noise channel acts on.
    probability: float
        The probability that this error occurs.
    """

    def __init__(
        self, qubit: Qubit[T] | T, probability: float, *, tag: str | None = None
    ):
        super().__init__(qubit, probability, tag=tag)

    def approx_equals(
        self, other: object, *, rel_tol: float = 1e-9, abs_tol: float = 0
    ) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.qubit == other.qubit
            and math.isclose(
                self.probability, other.probability, rel_tol=rel_tol, abs_tol=abs_tol
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.qubit == other.qubit
            and self.probability == other.probability
        )

    def __hash__(self) -> int:
        return hash((self.__class__, self.qubit, self.probability))

    def __repr__(self) -> str:
        tag_repr = f"[{self._tag}]" if self._tag is not None else ""
        return (
            f"{self.stim_string}{tag_repr}({self.qubit}, "
            f"probability={self.probability})"
        )


class PauliProductNoise(OneProbabilityNoiseChannel[T]):
    """Abstract noise channel which takes a Pauli product as argument.

    Attributes
    ----------
    stim_string: str
        The string that stim associates to this gate.

    Parameters
    ----------
    pauli_product: PauliGateT |
                   Iterable[PauliGateT] |
                   PauliProduct[T]
        The Pauli product to enact this noise channel
    probability: float
        The probability that this noise channel is applied
    """

    def __init__(
        self,
        pauli_product: _PauliGate | Iterable[_PauliGate] | PauliProduct[T],
        probability: float,
        tag: str | None = None,
    ):
        super().__init__(probability, tag=tag)
        self._pauli_product = (
            pauli_product
            if isinstance(pauli_product, PauliProduct)
            else PauliProduct(pauli_product)
        )

    @classmethod
    def generator_from_prob(
        cls, pauli_gate_t: type[_PauliGate], probability: float, tag: str | None = None
    ) -> Callable[[Sequence[Qubit[T] | T]], Sequence[Self]]:
        """Return a classmethod that can be used to create a noise channel
        with a predetermined probability"""

        def inner_gen(qubits: Sequence[Qubit[T] | T]) -> list[Self]:
            return [
                cls(
                    PauliProduct([pauli_gate_t(qubit) for qubit in qubits]),
                    probability,
                    tag=tag,
                )
            ]

        return inner_gen

    @property
    def pauli_product(self) -> PauliProduct[T]:
        """Get the Pauli product that defines this noise channel."""
        return self._pauli_product

    @property
    def qubits(self) -> tuple[Qubit[T], ...]:
        return self.pauli_product.qubits

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        self._pauli_product.transform_qubits(id_mapping)

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> tuple[stim.GateTarget, ...]:
        return self.pauli_product.stim_targets(qubit_mapping)

    def approx_equals(
        self, other: object, *, rel_tol: float = 1e-9, abs_tol: float = 0
    ) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.pauli_product == other.pauli_product
            and math.isclose(
                self.probability, other.probability, rel_tol=rel_tol, abs_tol=abs_tol
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.pauli_product == other.pauli_product
            and self.probability == other.probability
        )

    def __hash__(self) -> int:
        return hash((self._pauli_product, self._probability))

    def __repr__(self) -> str:
        tag_repr = f"[{self._tag}]" if self._tag is not None else ""
        return (
            f"{self.stim_string}{tag_repr}({self.pauli_product}, "
            f"probability={self.probability})"
        )
