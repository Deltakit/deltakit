# (c) Copyright Riverlane 2020-2025.
"""
This module contains common implementation parts for stabiliser codes.
CSSCode class derives from StabiliserCode.
"""

import itertools
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Sequence

from deltakit_circuit import Qubit
from deltakit_circuit._basic_types import PauliBasis
from deltakit_circuit._qubit_identifiers import PauliGate

from deltakit_explorer.codes._css._css_stage import CSSStage
from deltakit_explorer.codes._css._stabiliser_helper_functions import (
    pauli_gates_to_stim_pauli_string,
)
from deltakit_explorer.codes._stabiliser import Stabiliser


class StabiliserCode(ABC):
    """
    Class representing an abstract stabiliser code.

    This class also contains methods that help set up circuits for memory
    experiments.
    """

    def __init__(
        self,
        stabilisers: Sequence[Iterable[Stabiliser]],
        x_logical_operators: Sequence[Collection[PauliGate]] | None = None,
        z_logical_operators: Sequence[Collection[PauliGate]] | None = None,
        use_ancilla_qubits: bool = True,
        check_logical_operators_are_independent: bool = False,
    ):
        self._check_logical_operators_are_independent = (
            check_logical_operators_are_independent
        )
        self._stabilisers = tuple(
            tuple(simultaneous_stabilisers) for simultaneous_stabilisers in stabilisers
        )
        self._use_ancilla_qubits = use_ancilla_qubits
        if not hasattr(self, "_data_qubits"):
            self._data_qubits = self._calculate_data_qubits()
        self._ancilla_qubits = self._calculate_ancilla_qubits()

        if (x_logical_operators is None) != (z_logical_operators is None):
            msg = "Either both or neither of the logical operators should be provided."
            raise ValueError(msg)

        self._x_logical_operators = (
            tuple(frozenset(x_logical) for x_logical in x_logical_operators)
            if x_logical_operators is not None
            else None
        )
        self._z_logical_operators = (
            tuple(frozenset(z_logical) for z_logical in z_logical_operators)
            if z_logical_operators is not None
            else None
        )

    @abstractmethod
    def encode_logical_zeroes(self) -> CSSStage:
        r"""
        Set up the CSSStage which represents the encoding of a logical
        :math:`\ket{0}\dots\ket{0}`.

        Returns
        -------
        CSSStage
            CSSStage representing the encoding of a logical
            :math:`\ket{0}\dots\ket{0}`.
        """

    @abstractmethod
    def encode_logical_pluses(self) -> CSSStage:
        r"""
        Set up the CSSStage which represents the encoding of a logical
        :math:`\ket{+}\dots\ket{+}`

        Returns
        -------
        CSSStage
            CSSStage representing the encoding of a logical
            :math:`\ket{+}\dots\ket{+}`.
        """

    def _calculate_data_qubits(self) -> set[Qubit]:
        """
        Calculate all data qubits and return a set of them.

        Returns
        -------
        Set[Qubit]
            Set of all data qubits.
        """
        data_qubits: set[Qubit] = set()
        for simultaneous_stabilisers in self._stabilisers:
            for stab in simultaneous_stabilisers:
                data_qubits.update(stab.data_qubits)
        return data_qubits

    def _calculate_ancilla_qubits(self) -> set[Qubit]:
        """
        Calculate ancilla qubits.

        Returns
        -------
        Set[Qubit]
            Set of all ancilla qubits.
        """
        if not self._use_ancilla_qubits:
            return set()

        ancilla_attrs = set(
            itertools.chain.from_iterable(
                {stab.ancilla_qubit for stab in simultaneous_stabilisers}
                for simultaneous_stabilisers in self._stabilisers
            )
        )
        if None in ancilla_attrs:
            msg = (
                "In order to perform syndrome extraction using ancilla qubits, all "
                "the Stabilisers must have ancilla qubits defined."
            )
            raise ValueError(msg)
        return ancilla_attrs

    def measure_stabilisers(self, num_rounds: int) -> CSSStage:
        """
        Set up the CSSStage which represents measuring the stabilisers for num_rounds
        rounds.

        Parameters
        ----------
        num_rounds : int
            Number of rounds to measure the stabilisers for.

        Returns
        -------
        CSSStage
            CSSStage representing measuring the stabiliser for num_rounds
            rounds.
        """
        return CSSStage(
            num_rounds=num_rounds,
            stabilisers=self._stabilisers,
            use_ancilla_qubits=self._use_ancilla_qubits,
        )

    @abstractmethod
    def measure_z_logicals(self) -> CSSStage:
        """
        Set up the CSSStage which represents measuring all logical Z operators
        simultaneously.

        Returns
        -------
        CSSStage
             CSSStage representing measuring the logical Z.
        """

    @abstractmethod
    def measure_x_logicals(self) -> CSSStage:
        """
        Set up the CSSStage which represents measuring all logical X operators
        simultaneously.

        Returns
        -------
        CSSStage
             CSSStage representing measuring the logical X.
        """

    @property
    def stabilisers(self) -> tuple[tuple[Stabiliser, ...], ...]:
        """
        Code stabilisers.

        Returns
        -------
        Tuple[Tuple[Stabiliser, ...], ...]
            Code stabilisers.
        """
        return self._stabilisers

    @property
    def qubits(self) -> set[Qubit]:
        """
        All code qubits, both ancilla and data.

        Returns
        -------
        Set[Qubit]
            A set of qubits.
        """
        return self._ancilla_qubits.union(self._data_qubits)

    @property
    def data_qubits(self) -> set[Qubit]:
        """
        All code data qubits.

        Returns
        -------
        Set[Qubit]
            A set of data qubits.
        """
        return self._data_qubits

    @property
    def ancilla_qubits(self) -> set[Qubit]:
        """
        All code ancilla qubits.

        Returns
        -------
        Set[Qubit]
            A set of ancilla qubits.
        """
        return self._ancilla_qubits

    @property
    def use_ancilla_qubits(self) -> bool:
        """
        Whether ancilla qubits are used for syndrome extraction.

        Returns
        -------
        bool
            True, if ancillas are used.
        """
        return self._use_ancilla_qubits

    @property
    def x_logical_operators(self) -> tuple[frozenset[PauliGate], ...]:
        """
        All X logical operators for the code.

        Returns
        -------
        Tuple[FrozenSet[PauliGate]]
            A tuple of frozensets of `PauliGate`s, with each frozenset
            corresponding to one logical X operator.
        """
        return (
            self._x_logical_operators
            if self._x_logical_operators is not None
            else ()
        )

    @x_logical_operators.setter
    def x_logical_operators(self, new_x_logicals: Sequence[Collection[PauliGate]]):
        """
        Set the value of x_logical_operators with new Pauli-products.
        This setter will validate that each new logical:
        (a) Commutes with all the other new logicals,
        (b) Commutes with all stabilisers,
        (c) Anti-commute with exactly one Z-logical (at the same index), and commute with
        the others In the event that any check fails, an error is raised. It is also
        checked whether there are as many new logicals as existing logicals, and a
        ValueError is thrown if there is a difference.
        If the code's logicals are being set for the first time, and the code currently
        has no logicals, the above criteria are not checked, but will be checked for any
        subsequent setting of logicals.

        Parameters
        ----------
        new_x_logicals : Sequence[Collection[PauliGate]]
            New X-logicals, to be validated and set if valid.
        """
        self._validate_new_logicals(
            new_x_logicals,
            PauliBasis.Z,
        )
        self._x_logical_operators = tuple(frozenset(log) for log in new_x_logicals)

    @property
    def z_logical_operators(self) -> tuple[frozenset[PauliGate], ...]:
        """
        All Z logical operators for the code.

        Returns
        -------
        Tuple[FrozenSet[PauliGate]]
            A tuple of frozensets of `PauliGate`s, with each frozenset
            corresponding to one logical Z operator.
        """
        return (
            self._z_logical_operators
            if self._z_logical_operators is not None
            else ()
        )

    @z_logical_operators.setter
    def z_logical_operators(self, new_z_logicals: Sequence[Collection[PauliGate]]):
        """
        Set the value of z_logical_operators with new Pauli-products.
        This setter will validate that each new logical:
        (a) Commutes with all the other new logicals,
        (b) Commutes with all stabilisers,
        (c) Anti-commutes with exactly one Z-logical (at the same index), and commutes with
        the others.
        In the event that any check fails, an error is raised. It is also
        checked whether there are as many new logicals as existing logicals, and a
        ValueError is thrown if there is a difference.
        If the code's logicals are being set for the first time, and the code currently
        has no logicals, the above criteria are not checked, but will be checked for any
        subsequent setting of logicals.

        Parameters
        ----------
        new_z_logicals : Sequence[Collection[PauliGate]]
            New Z-logicals, to be validated and set if valid.
        """
        self._validate_new_logicals(
            new_z_logicals,
            PauliBasis.X,
        )
        self._z_logical_operators = tuple(frozenset(log) for log in new_z_logicals)

    def overwrite_logicals(
        self,
        new_x_logicals: Sequence[Collection[PauliGate]],
        new_z_logicals: Sequence[Collection[PauliGate]],
    ):
        """
        Method to allow for updating both X and Z logicals simultaneously. For instance,
        the user may want to define a new logical that does not anti-commute with the
        current other type logical, but that will anti-commute with the new logical to
        be specified. Trying to do this in sequence will raise an error, so this method
        allows both to be updated simultaneously.
        Validation is performed on all the new logical operators.

        Parameters
        ----------
        new_x_logicals : Sequence[Collection[PauliGate]]
            New X-logicals, to be validated and set if valid.
        new_z_logicals : Sequence[Collection[PauliGate]]
            New Z-logicals, to be validated and set if valid.
        """
        # Use private attribute to bypass validation for first setting of X
        self._x_logical_operators = tuple(frozenset(log) for log in new_x_logicals)
        # Set Z using the setter, to validate the new Z against the new X
        self.z_logical_operators = tuple(frozenset(log) for log in new_z_logicals)
        # Then also set X again, where it will now be compared to the Z that has been
        # set, as X could possibly have been invalid (e.g, not commute with stabilisers)
        self.x_logical_operators = self._x_logical_operators

    def _validate_new_logicals(
        self,
        new_logicals: Sequence[Collection[PauliGate]],
        opposite_logical_type: PauliBasis,
    ):
        """
        Given a sequence of new logicals, a code, stabilisers of the
        code and the opposite type logicals to the ones being validated, check whether the
        provided new logicals
        (a) Commutes with all the other new logicals,
        (b) Commutes with all stabilisers,
        (c) Anti-commutes with exactly one Z-logical (at the same index), and commutes with
        the others.
        In the event that any check fails, an error is raised. It is also
        checked whether there are as many new logicals as existing logicals, and a
        ValueError is thrown if there is a difference.
        If the code's logicals are being set for the first time, and the code currently
        has no logicals, the above criteria are not checked, but will be checked for any
        subsequent setting of logicals.

        Parameters
        ----------
        new_logicals : Sequence[Collection[PauliGate]]
            The new logical operators to validate.
        opposite_logical_type : PauliBasis
            The basis of the logical operators against which the new logicals will be
            checked for anti-commutativity.
        """
        stabilisers = self.stabilisers
        data_qubit_to_index = {qubit: i for i, qubit in enumerate(self.data_qubits)}
        opposite_type_logicals = (
            self.z_logical_operators
            if opposite_logical_type == PauliBasis.Z
            else self.x_logical_operators
        )
        # Allow there being 0 opposite logicals, in the case where a code is initialised without
        # logicals, then they are added later.
        if len(opposite_type_logicals) > 0 and len(new_logicals) != len(
            opposite_type_logicals
        ):
            msg = (
                "There must be as many new logicals as existing logicals,"
                f" but there are {len(new_logicals)}"
                f" while there should be {len(opposite_type_logicals)}"
            )
            raise ValueError(msg)
        if any(len(log) == 0 for log in new_logicals):
            msg = "Logicals cannot be weight 0"
            raise ValueError(msg)

        # Convert stabilisers and logicals to stim.PauliString for commutation checks
        stabilisers_as_pauli_strings = [
            pauli_gates_to_stim_pauli_string(stab.paulis, data_qubit_to_index)
            for stab in itertools.chain.from_iterable(stabilisers)
        ]
        new_logs_as_pauli_string = [
            pauli_gates_to_stim_pauli_string(logical, data_qubit_to_index)
            for logical in new_logicals
        ]
        opposite_logs_as_pauli_string = [
            pauli_gates_to_stim_pauli_string(logical, data_qubit_to_index)
            for logical in opposite_type_logicals
        ]

        for i, new_log_as_pauli_string in enumerate(new_logs_as_pauli_string):
            # Check logical commutes with other logicals of same type
            if not all(
                new_log_as_pauli_string.commutes(other_new_logical)
                for other_new_logical in new_logs_as_pauli_string[i + 1 :]
            ):
                msg = f"New logical at index {i} anti-commutes with other new logicals"
                raise ValueError(msg)

            # Check logical commutes with all stabilisers
            if not all(
                new_log_as_pauli_string.commutes(stab)
                for stab in stabilisers_as_pauli_strings
            ):
                msg = f"New logical at index {i} anti-commutes with stabilisers"
                raise ValueError(msg)

            # Check for anti-commutation if this isn't the first time setting logicals
            if (
                len(opposite_type_logicals) > 0
                and self._check_logical_operators_are_independent
            ):
                # Check new logical at index i anti-commutes with opposite-type logical
                # at index i
                if new_log_as_pauli_string.commutes(opposite_logs_as_pauli_string[i]):
                    msg = (
                        f"New logical at index {i} must anti-commute with opposite type"
                        f" logical at index {i}"
                    )
                    raise ValueError(msg)

                # Check new logical at index i commutes with all the other opposite-type
                # logical operators
                if not all(
                    new_log_as_pauli_string.commutes(log)
                    for log in opposite_logs_as_pauli_string[:i]
                    + opposite_logs_as_pauli_string[(i + 1) :]
                ):
                    msg = (
                        f"New logical at index {i} anti-commutes with opposite-type"
                        f" logicals other than at index {i}"
                    )
                    raise ValueError(msg)
