# (c) Copyright Riverlane 2020-2025.

"""
This module contains a class representing a stabiliser of a stabiliser code.
"""

from __future__ import annotations

from collections.abc import Collection

from deltakit_circuit import Qubit
from deltakit_circuit._qubit_identifiers import PauliGate, T


class Stabiliser:
    """
    Class representing a stabiliser.

    Parameters
    ----------
    paulis: Collection[PauliGate | None]
        The elementary Pauli terms in a stabiliser. If the input is ordered,
        this order specifies the schedule for syndrome extraction and
        None indicates an empty layer in the syndrome extraction circuit.
        A basic syndrome extraction circuit, using RX, CX, CY, CZ and MX
        gates, is constructed in the Stage class. For instance, `[PauliX(0),
        None, PauliY(1)]` corresponds to a syndrome extraction circuit with layers:

        - RX ancilla,
        - CX ancilla --> data qubit 0,
        - empty layer,
        - CY ancilla --> data qubit 1,
        - MX ancilla.

        Compilation to other gates is done in QPU class.
    ancilla_qubit: Optional[Qubit[T] | T]
        Ancilla qubit or None. Can be specified either directly as a Qubit or as an
        identifier (e.g., int). In the None case, we implicitly mean that
        the object doesn't have an associated syndrome extraction circuit.
        By default, it is None.

    Attributes
    ----------
    data_qubits: Set[Qubit]
        Set of all data qubits on which the stabiliser is defined.
    """

    def __init__(
        self,
        paulis: Collection[PauliGate | None],
        ancilla_qubit: Qubit[T] | T | None = None,
    ):
        """
        Raises
        ------
        ValueError
            If the ancilla qubit also acts as a data qubit and the stabiliser
            operator consists of more than one Pauli.
        """
        if isinstance(ancilla_qubit, Qubit):
            self.ancilla_qubit: Qubit[T] = ancilla_qubit
        elif ancilla_qubit is None:
            self.ancilla_qubit = ancilla_qubit
        else:
            self.ancilla_qubit = Qubit(ancilla_qubit)

        self._check_data_qubits([pauli.qubit for pauli in paulis if pauli is not None])
        self.data_qubits: set[Qubit] = {
            pauli.qubit for pauli in paulis if pauli is not None
        }

        if self.ancilla_qubit in self.data_qubits and len(paulis) > 1:
            # For now we accept the ancilla being the data_qubit, if we have only
            # one pauli in paulis. This is useful when putting together stages into
            # an experiment.
            msg = "Ancilla qubit should be different from the data qubits."
            raise ValueError(msg)

        self.paulis: tuple[PauliGate | None, ...] = tuple(paulis)

    @staticmethod
    def _check_data_qubits(qubits: Collection[Qubit]) -> None:
        """
        Check qubits contains only unique elements and is not empty.

        Raises
        ------
        ValueError
            If qubits contains duplicate elements.
        ValueError
            If qubits is empty.
        """
        if len(set(qubits)) == 0:
            msg = "Stabiliser was initialised without Pauli terms."
            raise ValueError(msg)
        if len(set(qubits)) != len(qubits):
            msg = "Data qubits given in paulis should be unique."
            raise ValueError(msg)

    @property
    def operator_repr(self) -> set[PauliGate]:
        """
        A set of PauliX/Y/Z's representing the Pauli operator. Useful
        when comparing Stabilisers.

        Returns
        -------
        Set[PauliGate]
            Pauli operator representation.
        """
        return {pauli for pauli in self.paulis if pauli is not None}

    def __eq__(self, other: object) -> bool:
        """
        Check if two Stabiliser objects have the same paulis and
        ancilla_qubit attributes.
        """
        return (
            isinstance(other, Stabiliser)
            and self.paulis == other.paulis
            and self.ancilla_qubit == other.ancilla_qubit
        )

    def __hash__(self) -> int:
        return hash((self.__class__, self.paulis, self.ancilla_qubit))

    def __mul__(self, other: Stabiliser) -> Stabiliser:
        """
        Multiply two Pauli products with disjoint supports, as operators. The
        returned object's ancilla_qubit is None!
        E.g. X_1 X_2 * Z_3 Y_4 will be X_1 X_2 Z_3 Y_4, but X_1 X_2 * Z_1 is not
        defined!

        Returns
        -------
        Stabiliser
            The product without ancilla.
        """
        if isinstance(other, Stabiliser) and self.data_qubits.isdisjoint(
            other.data_qubits
        ):
            return Stabiliser(
                paulis=[
                    pauli for pauli in self.paulis + other.paulis if pauli is not None
                ],
            )
        return NotImplemented

    def __repr__(self) -> str:
        return f"Stabiliser({self.paulis}, {self.ancilla_qubit})"
