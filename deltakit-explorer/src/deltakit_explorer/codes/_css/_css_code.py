# (c) Copyright Riverlane 2020-2025.
"""
This module defines a class for a CSS code. The class includes
multiple validity checks and provides methods to generate encoding,
stabiliser measurements, and logical measurement stages.
"""

from __future__ import annotations

import itertools
from collections import Counter
from functools import cached_property
from typing import (Collection, FrozenSet, Iterable, List, Optional, Sequence,
                    Set, Tuple)

import galois
import numpy as np
from deltakit_circuit import PauliX, PauliZ, Qubit
from deltakit_circuit._qubit_identifiers import PauliGate
from deltakit_circuit.gates import MX, MZ, RX, RZ
from deltakit_explorer.codes._css._css_stage import CSSStage
from deltakit_explorer.codes._css._stabiliser_code import StabiliserCode
from deltakit_explorer.codes._logicals import (
    get_logical_operators_from_css_parity_check_matrices,
    get_logical_operators_from_tableau)
from deltakit_explorer.codes._stabiliser import Stabiliser
from numpy.typing import NDArray


class CSSCode(StabiliserCode):
    """
    Class representing a CSS code.

    Parameters
    ----------
    stabilisers : Sequence[Sequence[Stabiliser]]
        Stabilisers defining the code. The i-th element of stabilisers contains an
        Iterable of stabilisers measured during the i-th time step. For instance:

        - if we intend to measure all stabilisers simultaneously, then this
          parameter could be simply given as [{...all stabilisers...}],
        - if we intend to do a spaced syndrome extraction, then this
          parameter could be simply given as
          [{...all x stabilisers...}, {...all z stabilisers...}].

        If any stabilisers do not have ancilla qubits, the `use_ancilla_qubits`
        parameter must be set to False.
    x_logical_operators : Optional[Sequence[Collection[PauliGate]]], optional
        An entry to provide manually the X logical operators of the CSS code. These
        operators must contain X Pauli terms only. If provided, then the Z logical
        operators must also be manually provided, otherwise a ValueError is raised.
        If `check_logical_operators_are_independent` is True, then each X logical operator
        must exactly only anticommute with the Z logical operator at the same index.
        There is also the possibility of automatically generating these logical operators,
        in which case neither should be provided and `calculate_logical_operators` should
        be set to True. To initialise with no logical operators, keep as None and set
        `calculate_logical_operators` to False.
        By default, None.
    z_logical_operators : Optional[Sequence[Collection[PauliGate]]], optional
        An entry to provide manually the Z logical operators of the CSS code. These
        operators must contain Z Pauli terms only. If provided, then the X logical
        operators must also be manually provided, otherwise a ValueError is raised.
        If `check_logical_operators_are_independent` is True, then each Z logical operator
        must exactly only anticommute with the X logical operator at the same index.
        There is also the possibility of automatically generating these logical operators,
        in which case neither should be provided and `calculate_logical_operators` should
        be set to True. To initialise with no logical operators, keep as None and set
        `calculate_logical_operators` to False.
        By default, None.
    use_ancilla_qubits : bool, optional
        Specifies whether ancilla qubits will be used for syndrome extraction. If
        this is True, the stabilisers must all have ancilla qubits and, in this case,
        the order the Paulis are given in the stabilisers will be the order in which
        information is extracted from the data qubits to the ancilla when performing
        syndrome extraction.
        By default, True.
    calculate_logical_operators: bool, optional,
        Whether to automatically calculate the logical operators. If this is True,
        then x_logical_operators and z_logical_operators should be None. Otherwise,
        a ValueError is raised.
        If `check_logical_operators_are_independent` is set to True, then the calculated
        logical operators will either be guaranteed to be independent or an error will
        be raised indicating that it didn't manage to meet this condition.
        By default, False.
    check_logical_operators_are_independent : bool, optional
        Whether to check if logical X-Z operator pairs are independent. If True, then
        the commuting relations [X_k, Z_l] = 0 (k != l) are checked for the logical
        operators, ensuring that the logical operators correspond to separate logical
        qubits. If False, this check is ignored, in which case the logical operators
        can be dependent, e.g. instead of only accepting X_1, X_2 and Z_1, Z_2, the
        input X_1, X_1*X_2 and Z_1, Z_1*Z_2 is also accepted.
        If both this flag and `calculate_logical_operators` are True, then the
        calculated logical operators will either be guaranteed to be independent or
        an error will be raised indicating that it didn't manage to meet this
        condition.
        By default, False.
    """

    def __init__(
        self,
        stabilisers: Sequence[Iterable[Stabiliser]],
        x_logical_operators: Optional[Sequence[Collection[PauliGate]]] = None,
        z_logical_operators: Optional[Sequence[Collection[PauliGate]]] = None,
        use_ancilla_qubits: bool = True,
        calculate_logical_operators: bool = False,
        check_logical_operators_are_independent: bool = False,
    ) -> None:
        super().__init__(
            stabilisers=stabilisers,
            x_logical_operators=x_logical_operators,
            z_logical_operators=z_logical_operators,
            use_ancilla_qubits=use_ancilla_qubits,
            check_logical_operators_are_independent=check_logical_operators_are_independent,
        )

        self._check_logical_operators_are_independent = (
            check_logical_operators_are_independent
        )
        self._data_qubit_index = {
            qubit: i
            for i, qubit in enumerate(
                sorted(self._data_qubits, key=lambda q: q.unique_identifier)
            )
        }

        self._x_stabilisers, self._z_stabilisers = self._calculate_x_and_z_stabilisers(
            self._stabilisers
        )

        if calculate_logical_operators:
            if (x_logical_operators is not None) or (z_logical_operators is not None):
                raise ValueError(
                    "No logicals should be provided if calculate_logical_operators is set to True."
                )
            (
                x_logical_operators,
                z_logical_operators,
            ) = self._calculate_logical_operators()
            self.overwrite_logicals(x_logical_operators, z_logical_operators)

        if getattr(self, "_perform_css_checks", True):
            self._check_duplicate_stabilisers(self._stabilisers)
            self._check_for_duplicate_paulis(
                x_logical_operators if x_logical_operators is not None else [],
                z_logical_operators if z_logical_operators is not None else [],
            )
            self._check_stabiliser_and_logical_operator_types(
                self._stabilisers, self.x_logical_operators, self.z_logical_operators
            )
            self._check_logical_operators_are_supported_on_data_qubits(
                self.x_logical_operators + self.z_logical_operators, self._data_qubits
            )
            self._check_ancilla_qubit_properties(
                self._ancilla_qubits, self._data_qubits, self._stabilisers
            )
            self._check_commutation_relations(
                self._x_stabilisers,
                self._z_stabilisers,
                self.x_logical_operators,
                self.z_logical_operators,
                self._check_logical_operators_are_independent,
            )
            if self._use_ancilla_qubits:
                self._check_stabiliser_lengths(self._stabilisers)
                self._check_unique_data_qubits_in_layers(self._stabilisers)
                self._check_schedule_is_valid(self._x_stabilisers, self._z_stabilisers)

    @staticmethod
    def x_and_z_operators_commute(
        x_operator: Iterable[Optional[PauliX]], z_operator: Iterable[Optional[PauliZ]]
    ) -> bool:
        """
        Compute whether an X-type stabiliser commutes with a Z-type stabiliser.
        """
        anticommutation_counter = Counter(
            (paulix.qubit == pauliz.qubit)
            for paulix in x_operator
            if paulix is not None
            for pauliz in z_operator
            if pauliz is not None
        )

        return (anticommutation_counter[True] % 2) == 0

    @staticmethod
    def _check_duplicate_stabilisers(
        stabilisers: Tuple[Tuple[Stabiliser, ...], ...]
    ) -> None:
        for ind_lay, simultaneous_stabilisers in enumerate(stabilisers):
            if len(simultaneous_stabilisers) != len(set(simultaneous_stabilisers)):
                raise ValueError(f"Layer {ind_lay} of stabilisers contains duplicates.")

    @staticmethod
    def _check_for_duplicate_paulis(
        x_logical_operators: Sequence[Collection[PauliGate]],
        z_logical_operators: Sequence[Collection[PauliGate]],
    ) -> None:
        """
        Check if there are any PauliGate duplicates in any of the logical operators.
        """
        for x_log_op in x_logical_operators:
            if len(set(x_log_op)) < len(x_log_op):
                raise ValueError(
                    "One of the X-logical operators contains duplicate PauliX objects."
                )
        for z_log_op in z_logical_operators:
            if len(set(z_log_op)) < len(z_log_op):
                raise ValueError(
                    "One of the Z-logical operators contains duplicate PauliZ objects."
                )

    @staticmethod
    def _check_stabiliser_and_logical_operator_types(
        stabilisers: Tuple[Tuple[Stabiliser, ...], ...],
        x_logical_operators: Tuple[FrozenSet[PauliGate], ...],
        z_logical_operators: Tuple[FrozenSet[PauliGate], ...],
    ) -> None:
        """
        Check the following:

            1. each stabiliser has either only X or only Z Pauli terms,

            2. the number of stabilisers is at least one,

            3. the lengths of x_logical_operators and z_logical_operators are equal,

            4. none of the logical operators is the identity,

            5. each X logical operator has only X Pauli terms, and

            6. each Z logical operator has only Z Pauli terms.
        """
        # Check condition 1)
        for simultaneous_stabilisers in stabilisers:
            for stabiliser in simultaneous_stabilisers:
                pauli_types = {
                    type(pauli) for pauli in stabiliser.paulis if pauli is not None
                }
                if len(pauli_types) != 1 or not pauli_types.issubset({PauliX, PauliZ}):
                    raise ValueError(
                        "CSSCode object was initialised with incorrect type "
                        "stabilisers. Each stabiliser must consist of either all X or "
                        "all Z Paulis terms."
                    )
        # Check condition 2)
        if (
            sum(
                len(simultaneous_stabilisers)
                for simultaneous_stabilisers in stabilisers
            )
            == 0
        ):
            raise ValueError("CSSCode object was initialised with no stabilisers.")
        # Check condition 3)
        if len(x_logical_operators) != len(z_logical_operators):
            raise ValueError(
                "The lengths of x_logical_operators and z_logical_operators are not "
                "equal."
            )
        # Check condition 4)
        if any(len(x_logical) == 0 for x_logical in x_logical_operators):
            raise ValueError(
                "x_logical_operators contains an empty Iterable. Identity cannot be a"
                " logical operator."
            )
        if any(len(z_logical) == 0 for z_logical in z_logical_operators):
            raise ValueError(
                "z_logical_operators contains an empty Iterable. Identity cannot be a"
                " logical operator."
            )
        # Check condition 5)
        for x_logical in x_logical_operators:
            if not all(isinstance(pauli, PauliX) for pauli in x_logical):
                raise ValueError(
                    "All X logical operators should consist of only X Pauli terms."
                )
        # Check condition 6)
        for z_logical in z_logical_operators:
            if not all(isinstance(pauli, PauliZ) for pauli in z_logical):
                raise ValueError(
                    "All Z logical operators should consist of only Z Pauli terms."
                )

    @staticmethod
    def _calculate_x_and_z_stabilisers(
        stabilisers: Tuple[Tuple[Stabiliser, ...], ...],
    ) -> Tuple[List[Set[Stabiliser]], List[Set[Stabiliser]]]:
        """
        Return two lists, one containing the X-stabilisers and the other the
        Z-stabilisers.

        Returns
        -------
        Tuple[List[Set[Stabiliser]], List[Set[Stabiliser]]]
            Tuple containing X-stabilisers and Z-stabilisers.
        """
        x_stabilisers: List[Set[Stabiliser]] = []
        z_stabilisers: List[Set[Stabiliser]] = []
        for simultaneous_stabilisers in stabilisers:
            x_layer = set()
            z_layer = set()
            for stabiliser in simultaneous_stabilisers:
                if isinstance(next(iter(stabiliser.operator_repr)), PauliX):
                    x_layer.add(stabiliser)
                else:
                    z_layer.add(stabiliser)
            x_stabilisers.append(x_layer)
            z_stabilisers.append(z_layer)
        return x_stabilisers, z_stabilisers

    @staticmethod
    def check_logical_operators_are_independent(
        x_logicals: Iterable[Iterable[PauliX]], z_logicals: Iterable[Iterable[PauliZ]]
    ):
        """
        Check that the logical operators are independent. In other words, check that
        each logical operator commutes with all other logical operators except for
        their pair (i.e. the logical operator of opposite type located at the same
        index). Raise a ValueError otherwise.
        """
        logicals_zipped = list(zip(x_logicals, z_logicals))
        for i, (x_logical, z_logical) in enumerate(logicals_zipped):
            if CSSCode.x_and_z_operators_commute(x_logical, z_logical):
                raise ValueError(
                    f"The X and Z logical operators with the same index should anticommute,"
                    f" but at index {i} found {x_logical} and {z_logical}."
                )

            for other_x_logical, other_z_logical in logicals_zipped[i + 1 :]:
                for logical, other_logical in (
                    (x_logical, other_z_logical),
                    (z_logical, other_x_logical),
                ):
                    if not CSSCode.x_and_z_operators_commute(logical, other_logical):
                        raise ValueError(
                            "Two logical operators with different indices should commute,"
                            f" but found operators {logical} and {other_logical}."
                        )

    @staticmethod
    def _check_commutation_relations(
        x_stabilisers: List[Set[Stabiliser]],
        z_stabilisers: List[Set[Stabiliser]],
        x_logical_operators: Tuple[FrozenSet[PauliGate], ...],
        z_logical_operators: Tuple[FrozenSet[PauliGate], ...],
        check_logical_operators_are_independent: bool,
    ) -> None:
        """
        Check the commutation-anticommutation relations between stabilisers and logical
        operators. More precisely, check that:

            1. all stabilisers commute with each other,

            2. each logical operator commutes with each stabiliser, and

            3. each logical operator commutes with each logical operator except one
                (i.e. the other type of logical operator on the same logical qubit).
        """
        x_stabilisers_flat = [stab for sim_stabs in x_stabilisers for stab in sim_stabs]
        z_stabilisers_flat = [stab for sim_stabs in z_stabilisers for stab in sim_stabs]

        # Check condition 1)
        for x_stab, z_stab in itertools.product(x_stabilisers_flat, z_stabilisers_flat):
            if not CSSCode.x_and_z_operators_commute(x_stab.paulis, z_stab.paulis):
                raise ValueError(
                    "CSSCode object was initialised with anticommuting "
                    "stabilisers. Namely, the X-stabiliser defined on data qubits "
                    f"{x_stab.data_qubits} anticommutes with the Z-stabiliser "
                    f"defined on data qubits {z_stab.data_qubits}."
                )
        # Check condition 2)
        for x_stab, z_log in itertools.product(x_stabilisers_flat, z_logical_operators):
            if not CSSCode.x_and_z_operators_commute(x_stab.paulis, z_log):
                raise ValueError(
                    "CSSCode object was initialised with a Z-logical operator that "
                    "anticommutes with a stabiliser. Namely, the "
                    f"X-stabiliser defined on data qubits {x_stab.data_qubits} "
                    f"anticommutes with the Z-logical operator {z_log}."
                )
        for z_stab, x_log in itertools.product(z_stabilisers_flat, x_logical_operators):
            if not CSSCode.x_and_z_operators_commute(z_stab.paulis, x_log):
                raise ValueError(
                    "CSSCode object was initialised with an X-logical operator that "
                    "anticommutes with a stabiliser. Namely, the "
                    f"Z-stabiliser defined on data qubits {z_stab.data_qubits} "
                    "anticommutes with the "
                    f"X-logical operator {x_log} "
                )
        # Check condition 3)
        for (x_ind, x_log), (z_ind, z_log) in itertools.product(
            enumerate(x_logical_operators), enumerate(z_logical_operators)
        ):
            if x_ind == z_ind and CSSCode.x_and_z_operators_commute(x_log, z_log):
                raise ValueError(
                    "CSSCode object was initialised with commuting X- and Z-logical "
                    f"operators defined on the logical qubit at index {x_ind}. "
                    f"The X-logical operator at index {x_ind} "
                    f"is {x_log}, and the Z-logical operator at index {z_ind} is "
                    f"{z_log}."
                )
            if (
                check_logical_operators_are_independent
                and x_ind != z_ind
                and not CSSCode.x_and_z_operators_commute(x_log, z_log)
            ):
                raise ValueError(
                    "CSSCode object was initialised with anticommuting X- and "
                    "Z-logical operators that act on different logical qubits at "
                    f"indices {x_ind, z_ind}. The X-logical operator at index {x_ind} "
                    f"is {x_log}, and the Z-logical operator at index {z_ind} is "
                    f"{z_log}."
                )

    @staticmethod
    def _check_ancilla_qubit_properties(
        ancilla_qubits: Set[Qubit],
        data_qubits: Set[Qubit],
        stabilisers: Tuple[Tuple[Stabiliser, ...], ...],
    ) -> None:
        """
        Check the following properties for ancilla qubits:

            1. data qubits are disjoint from ancilla qubits,

            2. each set of stabilisers measured simultaneously has no repeated
            ancilla qubits.
        """
        if len(ancilla_qubits) > 0:
            if not data_qubits.isdisjoint(ancilla_qubits):
                raise ValueError(
                    f"The following ancilla qubits are also used as data qubits: "
                    f"{data_qubits.intersection(ancilla_qubits)}"
                )
            for ind, simultaneous_stabilisers in enumerate(stabilisers):
                simultaneous_ancillas = {
                    stab.ancilla_qubit for stab in simultaneous_stabilisers
                }
                if len(simultaneous_ancillas) != len(simultaneous_stabilisers):
                    raise ValueError(
                        f"There are duplicate ancilla qubits in layer {ind} of "
                        "stabilisers."
                    )

    @staticmethod
    def _check_logical_operators_are_supported_on_data_qubits(
        logical_operators: Tuple[FrozenSet[PauliGate], ...], data_qubits: Set[Qubit]
    ) -> None:
        """
        Check that the logical operators are defined on the code's data qubits.
        """
        log_ops_qubits = set()
        for log_op in logical_operators:
            for pauli in log_op:
                log_ops_qubits.add(pauli.qubit)
        if not log_ops_qubits.issubset(data_qubits):
            raise ValueError(
                "Some logical operators are not supported on the CSS code's data "
                "qubits."
            )

    @staticmethod
    def _check_stabiliser_lengths(
        stabilisers: Tuple[Tuple[Stabiliser, ...], ...],
    ) -> None:
        """
        Assuming all stabilisers are measured using syndrome extraction circuits, check
        if each Set of the Sequence contains Stabilisers with paulis attribute of the
        same length. (It is, however, possible for different Sets to contain
        Stabilisers of different lengths.)
        """
        for ind, simultaneous_stabilisers in enumerate(stabilisers):
            # raise an error if the lengths of paulis in the current stabiliser layer
            # are not all the same
            if (
                len(
                    set(
                        len(stabiliser.paulis)
                        for stabiliser in simultaneous_stabilisers
                    )
                )
                > 1
            ):
                raise ValueError(
                    f"Layer {ind} of stabilisers contains two elements whose paulis "
                    "attributes are of different lengths."
                )

    @staticmethod
    def _check_unique_data_qubits_in_layers(
        stabilisers: Tuple[Tuple[Stabiliser, ...], ...],
    ):
        """
        Assuming all stabilisers are measured using syndrome extraction circuits, check
        that stabilisers is specified in such a way that during syndrome extraction
        none of the data qubits is used multiple times in the same layer.
        """
        for ind, simultaneous_stabilisers in enumerate(stabilisers):
            if len(simultaneous_stabilisers) == 0:
                continue
            current_paulis_len = len(next(iter(simultaneous_stabilisers)).paulis)
            for layer_ind in range(current_paulis_len):
                paulis_and_nones_in_layer = [
                    # navigate down "columns" of 2d array, where column=layer
                    stab.paulis[layer_ind]
                    for stab in simultaneous_stabilisers
                ]
                qubits_in_layer = [
                    pauli.qubit
                    for pauli in paulis_and_nones_in_layer
                    if pauli is not None
                ]
                if len(qubits_in_layer) != len(set(qubits_in_layer)):
                    raise ValueError(
                        f"Layer {ind} of stabilisers contains at least two Stabiliser "
                        f"objects that have the same qubit at index {layer_ind} of "
                        "their paulis attributes. This means that the syndrome "
                        "extraction circuit is invalid, as each layer should "
                        "contain unique qubits only."
                    )

    @staticmethod
    def _check_schedule_is_valid(
        x_stabilisers: List[Set[Stabiliser]], z_stabilisers: List[Set[Stabiliser]]
    ) -> None:
        """
        Assuming all stabilisers are measured using syndrome extraction circuits,
        check that the schedules define a valid syndrome extraction for the code. This
        assumes that _check_unique_data_qubits_in_layers already passed.
        """

        def _entangled_ancillas(x_stab: Stabiliser, z_stab: Stabiliser) -> bool:
            """
            Compute whether the schedule between an X- and Z-type stabiliser pair
            entangles the ancilla qubits or not, assuming that their paulis attributes
            have the same length. Return the bool for later reference.
            """
            x_ordered_qubits = [
                pauli.qubit if pauli is not None else None for pauli in x_stab.paulis
            ]
            z_ordered_qubits = [
                pauli.qubit if pauli is not None else None for pauli in z_stab.paulis
            ]
            # Check twist. Imagine we pull all CX gates before the CZ gates and each
            # time for a data qubit (target qubit) CX happens after CZ, we switch
            # is_twisted, as that means a CZ between the two ancillas has to be
            # applied.
            is_twisted = False
            for ind, qubit in enumerate(x_ordered_qubits):
                if qubit is None:
                    continue
                if qubit in z_ordered_qubits[:ind]:
                    is_twisted = not is_twisted
            return is_twisted

        twisted_info = []
        for x_simultaneous_stabilisers, z_simultaneous_stabilisers in zip(
            x_stabilisers, z_stabilisers
        ):
            twisted_info_layer = []
            for x_stab, z_stab in itertools.product(
                x_simultaneous_stabilisers, z_simultaneous_stabilisers
            ):
                if _entangled_ancillas(x_stab, z_stab):
                    twisted_info_layer.append(
                        [x_stab.ancilla_qubit, z_stab.ancilla_qubit]
                    )
            twisted_info.append(twisted_info_layer)

        if any(len(twisted_info_layer) > 0 for twisted_info_layer in twisted_info):
            # Build error string
            error_str = ""
            for layer_ind, twisted_info_layer in enumerate(twisted_info):
                if len(twisted_info_layer) > 0:
                    error_str += "\n" + " " * 4 + f"in layer {layer_ind} of stabilisers"
                for ancilla_pair in twisted_info_layer:
                    x_anc, z_anc = ancilla_pair
                    error_str += "\n" + " " * 8 + str(x_anc) + " and " + str(z_anc)

            raise ValueError(
                "The scheduling of the stabilisers doesn't give a valid syndrome "
                "extraction circuit. More precisely, the provided schedule entangles "
                "the following ancilla pairs:" + error_str
            )

    def encode_logical_zeroes(self) -> CSSStage:
        if len(self.z_logical_operators) == 0:
            raise NotImplementedError(
                "No logical Z operators provided so cannot prepare logical zeroes "
                "state."
            )
        return CSSStage(final_round_resets=[RZ(qubit) for qubit in self._data_qubits])

    def encode_logical_pluses(self) -> CSSStage:
        if len(self.x_logical_operators) == 0:
            raise NotImplementedError(
                "No logical X operators provided so cannot prepare logical pluses "
                "state."
            )
        return CSSStage(final_round_resets=[RX(qubit) for qubit in self._data_qubits])

    def measure_z_logicals(self) -> CSSStage:
        if len(self.z_logical_operators) == 0:
            raise NotImplementedError(
                "No logical Z operators provided so cannot measure logical Z operators."
            )
        return CSSStage(
            first_round_measurements=[MZ(qubit) for qubit in self._data_qubits],
            observable_definitions={
                ind: [pauli.qubit for pauli in z_logical]
                for ind, z_logical in enumerate(self.z_logical_operators)
            },
        )

    def measure_x_logicals(self) -> CSSStage:
        if len(self.x_logical_operators) == 0:
            raise NotImplementedError(
                "No logical X operators provided so cannot measure logical X operators."
            )
        return CSSStage(
            first_round_measurements=[MX(qubit) for qubit in self._data_qubits],
            observable_definitions={
                ind: [pauli.qubit for pauli in x_logical]
                for ind, x_logical in enumerate(self.x_logical_operators)
            },
        )

    @staticmethod
    def from_matrix(
        h_x: NDArray = np.zeros((0, 0)),
        h_z: NDArray = np.zeros((0, 0)),
        log_x_ops: NDArray = np.zeros((0, 0)),
        log_z_ops: NDArray = np.zeros((0, 0)),
    ) -> CSSCode:
        """
        Return a CSSCode object with stabilisers that have no ancilla qubits. The
        inputs must be 0-1 matrices.

        Parameters
        ----------
        h_x : NDArray, optional
            The check matrix (containing only 0 and 1) for X stabilisers where each
            row represents an X stabiliser. If an empty matrix, then this means the
            CSS code has no X stabilisers. By default, np.zeros((0,0)).
        h_z : NDArray, optional
            The check matrix (containing only 0 and 1) for Z stabilisers where each
            row represents a Z stabiliser. If an empty matrix, then this means the
            CSS code has no Z stabilisers. By default, np.zeros((0,0)).
        log_x_ops : NDArray, optional
            Matrix (containing only 0 and 1) representing the logical X operators
            where each row represents a logical X operator. If an empty matrix,
            then this means the CSS code was specified without X logical
            operators. By default, np.zeros((0,0)).
        log_z_ops : NDArray, optional
            Matrix (containing only 0 and 1) representing the logical Z operators
            where each row represents a logical Z operator. If an empty matrix,
            then this means the CSS code was specified without Z logical
            operators. By default, np.zeros((0,0)).

        Returns
        -------
        CSSCode
            The CSS code obtained from the matrix input.

        Raises
        ------
        ValueError
            If both matrices h_x and h_z are empty.
        ValueError
            If neither of the matrices h_x_and h_z is empty but they have different
            numbers of columns.
        ValueError
            If the numbers of rows differ for the matrices log_x_ops and log_z_ops.
        ValueError
            If either log_x_ops or log_z_ops has a different number of columns to
            h_x or h_z.
        ValueError
            If any of the inputs have elements that are different from 0 and 1.
        """
        num_x_stabs, num_qubits_x = np.shape(h_x)
        num_z_stabs, num_qubits_z = np.shape(h_z)

        # Size checks for h_x and h_z
        if num_qubits_x == 0 and num_qubits_z == 0:
            raise ValueError("The matrices h_x and h_z cannot both be empty.")
        if num_qubits_x > 0 and num_qubits_z > 0 and num_qubits_x != num_qubits_z:
            raise ValueError(
                "The matrices h_x and h_z need to have the same number of columns."
            )

        num_qubits = num_qubits_x if num_qubits_x > 0 else num_qubits_z
        num_x_log_ops, num_qubits_log_x = np.shape(log_x_ops)
        num_z_log_ops, num_qubits_log_z = np.shape(log_z_ops)

        # Size checks for log_x_ops and log_z_ops
        if num_x_log_ops != num_z_log_ops:
            raise ValueError(
                "The matrices log_x_ops and log_z_ops cannot have different numbers of "
                "rows."
            )
        if num_qubits_log_x not in [0, num_qubits]:
            raise ValueError(
                "The matrix log_x_ops has a different number of columns to h_x or h_z."
            )
        if num_qubits_log_z not in [0, num_qubits]:
            raise ValueError(
                "The matrix log_z_ops has a different number of columns to h_x or h_z."
            )

        def _check_and_return_entry(entry):
            if entry not in [0, 1]:
                raise ValueError(
                    "Some elements in the provided matrices are not 0 or 1."
                )
            return entry

        x_stabilisers = (
            []
            if num_x_stabs == 0 or num_qubits_x == 0
            else [
                Stabiliser(
                    paulis=[
                        PauliX(j)
                        for j in range(num_qubits)
                        if _check_and_return_entry(h_x[i, j]) == 1
                    ]
                )
                for i in range(num_x_stabs)
            ]
        )
        z_stabilisers = (
            []
            if num_z_stabs == 0 or num_qubits_z == 0
            else [
                Stabiliser(
                    paulis=[
                        PauliZ(j)
                        for j in range(num_qubits)
                        if _check_and_return_entry(h_z[i, j]) == 1
                    ]
                )
                for i in range(num_z_stabs)
            ]
        )
        x_logical_operators = (
            []
            if num_x_log_ops == 0 or num_qubits_log_x == 0
            else [
                [
                    PauliX(j)
                    for j in range(num_qubits)
                    if _check_and_return_entry(log_x_ops[i, j]) == 1
                ]
                for i in range(num_x_log_ops)
            ]
        )
        z_logical_operators = (
            []
            if num_z_log_ops == 0 or num_qubits_log_z == 0
            else [
                [
                    PauliZ(j)
                    for j in range(num_qubits)
                    if _check_and_return_entry(log_z_ops[i, j]) == 1
                ]
                for i in range(num_z_log_ops)
            ]
        )
        return CSSCode(
            stabilisers=[x_stabilisers + z_stabilisers],
            x_logical_operators=x_logical_operators,
            z_logical_operators=z_logical_operators,
            use_ancilla_qubits=False,
        )

    @cached_property
    def parity_check_matrices(self) -> tuple[NDArray, NDArray]:
        """
        Construct parity check matrices Hx and Hz from the CSS Code.

        Note that neither logical operators nor ancilla qubits are contained in
        the parity check matrices.

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple of two matrices, Hx and Hz.
        """
        # first, assign an index to each data qubit which will be the index
        # of the column in the matrix. qubits may have strange coordinates or be
        # transformed, so we cannot rely on the qubit coordinates
        all_x_stabilisers = list(itertools.chain.from_iterable(self._x_stabilisers))
        all_z_stabilisers = list(itertools.chain.from_iterable(self._z_stabilisers))
        x_parity_mat, z_parity_mat = np.zeros(
            (len(all_x_stabilisers), len(self.data_qubits)), dtype=np.uint8
        ), np.zeros((len(all_z_stabilisers), len(self.data_qubits)), dtype=np.uint8)

        # turn stabilisers into rows of the matrices.
        # note that since stabilisers are kept in Sets, the order
        # in which we iterate over them is non-deterministic, and so
        # is the order of the rows of the parity check matrices
        for matrix, stab_type in zip(
            (x_parity_mat, z_parity_mat), (all_x_stabilisers, all_z_stabilisers)
        ):
            for stab_index, stab in enumerate(stab_type):
                for pauli in (p for p in stab.paulis if p is not None):
                    matrix[stab_index, self._data_qubit_index[pauli.qubit]] = 1

        return x_parity_mat, z_parity_mat

    def _calculate_logical_operators(
        self,
    ) -> tuple[tuple[set[PauliX], ...], tuple[set[PauliZ], ...]]:
        """
        Compute the logical operators of the code. First tries doing so by completing
        a stim Tableau, and if it doesn't yield X logicals purely made of X terms
        and Z logicals purely made of Z terms, falls back to doing so via the parity
        check matrix, for which that condition is guaranteed (but not necessarily
        independence).

        Tries to return independent logical operators but doesn't check whether they
        are.

        Returns
        -------
        tuple[tuple[set[PauliX], ...], tuple[set[PauliZ], ...]]
            The logical operators, provided as a tuple of all the X logical
            operators at index 0 and all the Z logical operators at index 1.
        """
        # First try computing from Tableau (guarantees independent logical operators).
        x_logicals, z_logicals = get_logical_operators_from_tableau(
            stabilisers=[
                stabiliser
                for simultaneous_stabilisers in self._stabilisers
                for stabiliser in simultaneous_stabilisers
            ],
            num_logical_qubits=self.calculate_number_of_logical_qubits(),
        )

        # Check X logical operators are made only of X gates and Z logical operators are
        # made only of Z gates (assumed by CSSCode).
        x_logicals_are_x_only = all(
            isinstance(pauli, PauliX) for logical in x_logicals for pauli in logical
        )
        z_logicals_are_z_only = all(
            isinstance(pauli, PauliZ) for logical in z_logicals for pauli in logical
        )

        if x_logicals_are_x_only and z_logicals_are_z_only:
            return x_logicals, z_logicals

        # Otherwise, fallback to method from BPOSD package, which guarantees X/Z logicals are X/Z-only
        # but not independence.
        x_logicals, z_logicals = get_logical_operators_from_css_parity_check_matrices(
            *self.parity_check_matrices,
            column_to_qubit={v: k for k, v in self._data_qubit_index.items()},
        )

        return x_logicals, z_logicals

    def calculate_number_of_logical_qubits(self) -> int:
        """
        Calculate the maximum number of logical qubits that are encoded into the code by
        subtracting the rank of Hx and Hz parity matrices from
        the number of (data) qubits.

        Note: this might not agree with the number of logicals that the user specified
        when initialising the `CSSCode` instance.
        """
        hx_mat, hz_mat = self.parity_check_matrices

        return len(self.data_qubits) - int(
            np.linalg.matrix_rank(galois.GF2(hx_mat))
            + np.linalg.matrix_rank(galois.GF2(hz_mat))
        )
