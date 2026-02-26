# (c) Copyright Riverlane 2020-2025.
"""Module which provides all measurement gates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import ClassVar, get_args

import stim

from deltakit_circuit._qubit_identifiers import (
    MeasurementPauliProduct,
    Qubit,
    T,
    U,
    _InvertiblePauliGate,
    _PauliGate,
)
from deltakit_circuit.gates._abstract_gates import (
    Gate,
    OneQubitMeasurementGate,
    PauliBasis,
)


class MZ(OneQubitMeasurementGate[T]):
    r"""Z-basis measurement (optionally noisy). Projects each target qubit
    into :math:`\ket{0}` or :math:`\ket{1}` and reports its value (false =
    :math:`\ket{0}`, true = :math:`\ket{1}`).

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.
    probability: float, optional
        A single float specifying the probability of flipping each reported
        measurement result, by default 0.0.
    invert: bool, optional
        Whether to invert the result of this measurement.

    Notes
    -----
    If this gate is parameterized by a probability argument, the recorded
    result will be flipped with that probability. If not, the recorded result
    is noiseless. Note that the noise only affects the recorded result, not
    the target qubit's state.

    | Stabilizer Generators:
    |   Z -> m xor chance(p)
    |   Z -> +Z

    | Decomposition (into H, S, CX, M, R):
    | The following circuit is equivalent (up to global phase) to ``M 0``
    |   M 0
    | (The decomposition is trivial because this gate is in the target gate
    | set.)
    """

    # pylint: disable=invalid-name

    basis: ClassVar[PauliBasis] = PauliBasis.Z
    stim_string: ClassVar[str] = f"M{basis.value}"


class MRZ(OneQubitMeasurementGate[T]):
    r"""Z-basis demolition measurement (optionally noisy). Projects each target
    qubit into :math:`\ket{0}` or :math:`\ket{1}`, reports its value (false =
    :math:`\ket{0}`, true = :math:`\ket{1}`), then resets to :math:`\ket{0}`.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.
    probability: float, optional
        A single float specifying the probability of flipping each reported
        measurement result, by default 0.0.
    invert: bool, optional
        Whether to invert the result of this measurement.

    Notes
    -----
    If this gate is parameterized by a probability argument, the recorded
    result will be flipped with that probability. If not, the recorded result
    is noiseless. Note that the noise only affects the recorded result, not
    the target qubit's state.

    | Stabilizer Generators:
    |   Z -> m xor chance(p)
    |   1 -> +Z

    | Decomposition (into H, S, CX, M, R):
    | The following circuit is equivalent (up to global phase) to ``MR 0``
    |   M 0
    |   R 0
    """

    basis: ClassVar[PauliBasis] = PauliBasis.Z
    stim_string: ClassVar[str] = f"MR{basis.value}"


class MRX(OneQubitMeasurementGate[T]):
    r"""X-basis demolition measurement (optionally noisy). Projects each target
    qubit into :math:`\ket{+}` or :math:`\ket{-}`, reports its value (false =
    :math:`\ket{+}`, true = :math:`\ket{-}`), then resets to :math:`\ket{+}`.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.
    probability: float, optional
        A single float specifying the probability of flipping each reported
        measurement result, by default 0.0.
    invert: bool, optional
        Whether to invert the result of this measurement.

    Notes
    -----
    If this gate is parameterized by a probability argument, the recorded
    result will be flipped with that probability. If not, the recorded result
    is noiseless. Note that the noise only affects the recorded result, not
    the target qubit's state.

    | Stabilizer Generators:
    |   X -> m xor chance(p)
    |   1 -> +X

    | Decomposition (into H, S, CX, M, R):
    | The following circuit is equivalent (up to global phase) to ``MRX 0``
    |   H 0
    |   M 0
    |   R 0
    |   H 0
    """

    basis: ClassVar[PauliBasis] = PauliBasis.X
    stim_string: ClassVar[str] = f"MR{basis.value}"


class MRY(OneQubitMeasurementGate[T]):
    r"""Y-basis demolition measurement (optionally noisy). Projects each target
    qubit into :math:`\ket{i}` or :math:`\ket{-i}`, reports its value (false =
    :math:`\ket{i}`, true = :math:`\ket{-i}`), then resets to :math:`\ket{i}`.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.
    probability: float, optional
        A single float specifying the probability of flipping each reported
        measurement result, by default 0.0.
    invert: bool, optional
        Whether to invert the result of this measurement.

    Notes
    -----
    If this gate is parameterized by a probability argument, the recorded
    result will be flipped with that probability. If not, the recorded result
    is noiseless. Note that the noise only affects the recorded result, not
    the target qubit's state.

    | Stabilizer Generators:
    |   Y -> m xor chance(p)
    |   1 -> +Y

    | Decomposition (into H, S, CX, M, R):
    | The following circuit is equivalent (up to global phase) to ``MRY 0``
    |   S 0
    |   S 0
    |   S 0
    |   H 0
    |   R 0
    |   M 0
    |   H 0
    |   S 0
    """

    basis: ClassVar[PauliBasis] = PauliBasis.Y
    stim_string: ClassVar[str] = f"MR{basis.value}"


class MX(OneQubitMeasurementGate[T]):
    r"""X-basis measurement (optionally noisy). Projects each target qubit into
     :math:`\ket{+}` or :math:`\ket{-}` and reports its value (false =
     :math:`\ket{+}`, true = :math:`\ket{-}`).

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.
    probability: float, optional
        A single float specifying the probability of flipping each reported
        measurement result, by default 0.0.
    invert: bool, optional
        Whether to invert the result of this measurement.

    Notes
    -----
    If this gate is parameterized by a probability argument, the recorded
    result will be flipped with that probability. If not, the recorded result
    is noiseless. Note that the noise only affects the recorded result, not
    the target qubit's state.

    | Stabilizer Generators:
    |   X -> +m xor chance(p)
    |   X -> +X

    | Decomposition (into H, S, CX, M, R):
    | The following circuit is equivalent (up to global phase) to ``MX 0``
    |   H 0
    |   M 0
    |   H 0
    """

    basis: ClassVar[PauliBasis] = PauliBasis.X
    stim_string: ClassVar[str] = f"M{basis.value}"


class MY(OneQubitMeasurementGate[T]):
    r"""Y-basis measurement (optionally noisy). Projects each target qubit into
    :math:`\ket{i}` or :math:`\ket{-i}` and reports its value (false =
    :math:`\ket{i}`, true = :math:`\ket{-i}`).

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.
    probability: float, optional
        A single float specifying the probability of flipping each reported
        measurement result, by default 0.0.
    invert: bool, optional
        Whether to invert the result of this measurement.

    Notes
    -----
    If this gate is parameterized by a probability argument, the recorded
    result will be flipped with that probability. If not, the recorded result
    is noiseless. Note that the noise only affects the recorded result, not
    the target qubit's state.

    | Stabilizer Generators:
    |   Y -> m xor chance(p)
    |   Y -> +Y

    | Decomposition (into H, S, CX, M, R):
    | The following circuit is equivalent (up to global phase) to ``MY 0``
    |   S 0
    |   S 0
    |   S 0
    |   H 0
    |   M 0
    |   H 0
    |   S 0
    """

    basis: ClassVar[PauliBasis] = PauliBasis.Y
    stim_string: ClassVar[str] = f"M{basis.value}"


# pylint: disable=invalid-name
class HERALD_LEAKAGE_EVENT(OneQubitMeasurementGate[T]):
    """
    The single qubit leakage heralding gate (optionally noisy).

    Determines whether or not a qubit is leaked and populates the measurement
    record with this data.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit that this gate acts on.
    probability: float, optional
        A single float specifying the probability of flipping each reported
        leaked measurement result, by default 0.0.
    invert: bool, optional
        Whether to invert the result of this measurement.

    Targets:
    --------
      Qubits to herald leakage for.

    Examples:
    ---------
      # Herald leakage on qubit 0. If 0 is leaked put a 1 in the measurement
      # record, otherwise a 0
      HERALD_LEAKAGE_EVENT 0

      # Same as above but heralds leakage on both qubit 0 and 1
      HERALD_LEAKAGE_EVENT 0 1

      # To populate a syndrome with the heralding bit during detector sampling
      # use a DETECTOR
      # annotation that refers back to this channel
      HERALD_LEAKAGE_EVENT 0
      DETECTOR rec[-1]
    """

    basis: ClassVar[PauliBasis | None] = None
    stim_string: ClassVar[str] = "HERALD_LEAKAGE_EVENT"


class MPP(Gate[T]):
    """Measure Pauli products.

    Parameters
    ----------
    pauli_product : Union[PauliGateT, InvertiblePauliGateT, \
                          Iterable[PauliGateT | InvertiblePauliGateT], \
                          MeasurementPauliProduct[T]]
        The product of pauli gates to measure. All options will internally be
        converted to a ``MeasurementPauliProduct`` instance.
    probability : float, optional
        A single float specifying the probability of flipping each reported
        measurement result, by default 0.0.

    Examples
    --------
    >>> import deltakit_circuit as sp
    >>> sp.gates.MPP([sp.PauliX(1), sp.PauliY(2)])
    MPP([PauliX(Qubit(1)), PauliY(Qubit(2))], probability=0.0)
    >>> sp.gates.MPP(~sp.InvertiblePauliZ(5))
    MPP([!PauliZ(Qubit(5))], probability=0.0)
    >>> sp.gates.MPP([sp.PauliX(1), sp.PauliX(2)], 0.01)
    MPP([PauliX(Qubit(1)), PauliX(Qubit(2))], probability=0.01)

    Notes
    -----
    If this gate is parameterized by a probability argument, the recorded
    result will be flipped with that probability. If not, the recorded result
    is noiseless. Note that the noise only affects the recorded result, not
    the target qubit's state.

    | Stabilizer Generators:
    |   P -> m xor chance(p)
    |   P -> P
    """

    stim_string: ClassVar[str] = "MPP"

    def __init__(
        self,
        pauli_product: (
            _PauliGate
            | _InvertiblePauliGate
            | Iterable[_PauliGate | _InvertiblePauliGate]
            | MeasurementPauliProduct[T]
        ),
        probability: float = 0.0,
        *,
        tag: str | None = None,
    ):
        super().__init__(tag=tag)
        if not 0 <= probability <= 1:
            msg = "Probability must be between zero and one."
            raise ValueError(msg)
        self._probability = probability
        self._pauli_product = (
            pauli_product
            if isinstance(pauli_product, MeasurementPauliProduct)
            else MeasurementPauliProduct(pauli_product)
        )

    @property
    def pauli_product(self) -> MeasurementPauliProduct:
        """Get the Pauli product for this gate."""
        return self._pauli_product

    @property
    def qubits(self) -> tuple[Qubit[T], ...]:
        """Get all qubits for this gate in a tuple."""
        return self.pauli_product.qubits

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        self._pauli_product.transform_qubits(id_mapping)

    def stim_targets(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> tuple[stim.GateTarget, ...]:
        """Get all qubits for this gate in a tuple."""
        return self.pauli_product.stim_targets(qubit_mapping)

    @property
    def probability(self) -> float:
        """Get the probability of flipping this measurement result."""
        return self._probability

    def approx_equals(
        self, other: object, *, rel_tol: float = 1e-9, abs_tol: float = 0
    ) -> bool:
        """Determines whether two MPP gates are approximately equal within a
        tolerance. The tolerance accounts for differences between the
        probabilities of the two MPP gates. All other properties must be equal.

        Parameters
        ----------
        other : object
            The other object to compare this MPP gate to.
        rel_tol : float
            The allowed relative difference between the error probabilities of
            the two MPP gates, if this is larger than that calculated from
            abs_tol. Has the same meaning as in math.isclose.
            By default, 1e-9.
        abs_tol : float, optional
            The allowed absolute difference between the error probabilities
            of the two MPP gates, if this is larger than that calculated
            from rel_tol. Has the same meaning as in math.isclose.
            By default, 0.0.

        Returns
        -------
        bool
        """
        return (
            isinstance(other, MPP)
            and self.pauli_product == other.pauli_product
            and math.isclose(
                self.probability, other.probability, rel_tol=rel_tol, abs_tol=abs_tol
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MPP)
            and self.pauli_product == other.pauli_product
            and self.probability == other.probability
        )

    def __hash__(self) -> int:
        return hash((self.__class__, self._pauli_product, self._probability))

    def __repr__(self) -> str:
        tag_repr = f"[{self._tag}]" if self._tag is not None else ""
        return (
            f"{self.stim_string}{tag_repr}({self.pauli_product}, "
            f"probability={self.probability})"
        )


_OneQubitMeasurementGate = MX | MY | MZ | MRX | MRY | MRZ | HERALD_LEAKAGE_EVENT
_MeasurementGate = _OneQubitMeasurementGate | MPP

MEASUREMENT_GATES: frozenset[type[_MeasurementGate]] = frozenset(
    get_args(_MeasurementGate)
)
ONE_QUBIT_MEASUREMENT_GATES = set(MEASUREMENT_GATES) - {MPP}
