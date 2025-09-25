# (c) Copyright Riverlane 2020-2025.
"""
This module defines a function to compute the logical operators associated
with a collection of stabilisers.
"""

from __future__ import annotations

import warnings
from typing import Collection, Iterable, Optional

from deltakit_circuit import PauliX, PauliY, PauliZ, Qubit
from deltakit_circuit._qubit_identifiers import _PauliGate
from deltakit_explorer.codes._css._stabiliser_helper_functions import \
    pauli_gates_to_stim_pauli_string
from deltakit_explorer.codes._stabiliser import Stabiliser
from numpy.typing import NDArray
from stim import PauliString, Tableau

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=SyntaxWarning)
    from bposd.css import css_code  # type: ignore


def paulistring_to_operator(
    paulistr: PauliString, index_to_qubit: dict[int, Qubit]
) -> list[_PauliGate]:
    """
    Converts a stim PauliString to a list of PauliGate objects.

    Parameters
    ----------
    paulistr : stim.PauliString
        The stim pauli string.
    index_to_qubit : dict[int, Qubit]
        A mapping from index in the pauli string to `deltakit.circuit.Qubit` object.

    Returns
    -------
    list[_PauliGate]
        The paulistring as a list of PauliGate objects.
    """
    return [
        (PauliX if el == 1 else PauliZ if el == 3 else PauliY)(index_to_qubit[el_index])
        for el_index, el in enumerate(paulistr)
        if el > 0
    ]


def get_str_logical_operators_from_tableau(
    stabilisers: Collection[PauliString], num_logical_qubits: Optional[int] = None
) -> list[tuple[PauliString, PauliString]]:
    """
    For a general stabiliser code, computes the logical operators for a collection of
    stabilisers.

    This method of computing the logical operators:
    - Guarantees the logical operators are independent,
    - Does NOT guarantee the logical operators are minimum-weight,
    - Does NOT guarantee for CSS codes that X logical operators are made purely of X gates
    and Z logical operators are made purely of Z gates (this is suspected but
    wasn't decidedly shown).

    From Stack Exchange post:
    https://quantumcomputing.stackexchange.com/questions/37812/how-to-find-a-set-of-independent-logical-operators-for-a-stabilizer-code-with-st

    Explanation from post:
    Solves for the observables as part of completing a tableau. It works by finding
    operations that turn the stabilisers into single-qubit terms. The observables are
    then created by looking at what undoing those operations turns the other qubits
    into.

    The stabilisers are provided as stim PauliStrings and so are the operators returned.

    Parameters
    ----------
    stabilisers : Collection[PauliString]
        The stabilisers as stim pauli string objects.
    num_logical_qubits : int, optional
        The number of logical qubits these stabilisers are expected to have. If provided,
        exactly this number of elements will be extracted from the end of the completed
        Tableau. If the number of logical qubits is known, providing it is safer
        because we are not 100% sure whether it is guaranteed that the stabilisers in
        the Tableau are exactly the same as the ones inputted (or whether they can be
        linear combinations of them, in which case the algorithm will think it is a
        logical operator). No example has been found where this was an issue so far, but
        this is to be extra safe. By default, None.

    Returns
    -------
    list[tuple[PauliString, PauliString]]
        The logical operators for the stabilisers provided. Each
        element in the list returned is an anticommuting pair
        of X and Z logical operators. Every other pair across and
        among commutes.
    """
    completed_tableau = Tableau.from_stabilizers(
        stabilisers,
        allow_redundant=True,
        allow_underconstrained=True,
    )

    iteration_range = range(len(completed_tableau))[::-1]

    if num_logical_qubits is not None:
        iteration_range = iteration_range[:num_logical_qubits]

    operators: list[tuple[PauliString, PauliString]] = []
    for k in iteration_range:
        z = completed_tableau.z_output(k)
        if z in stabilisers:
            break
        x = completed_tableau.x_output(k)
        operators.append((x, z))

    return operators


def get_logical_operators_from_tableau(
    stabilisers: Iterable[Stabiliser], num_logical_qubits: Optional[int] = None
) -> tuple[tuple[set[_PauliGate], ...], tuple[set[_PauliGate], ...]]:
    """
    For a general stabiliser code, computes the logical operators for a collection of
    stabilisers.

    This method of computing the logical operators:
    - Guarantees the logical operators are independent,
    - Does NOT guarantee the logical operators are minimum-weight,
    - Does NOT guarantee for CSS codes that X logical operators are made purely of X gates
    and Z logical operators are made purely purely of Z gates (this is suspected but
    wasn't decidedly shown).

    The stabilisers are provided as Stabiliser objects and the operators returned are
    made of Pauli gates.

    Parameters
    ----------
    stabilisers : Iterable[Stabiliser]
        The stabilisers from which to generate logical operators.
    num_logical_qubits : int, optional
        The number of logical qubits these stabilisers are expected to have. If provided,
        exactly this number of elements will be extracted from the end of the completed
        Tableau. If the number of logical qubits is known, providing it is safer
        because we are not 100% sure whether it is guaranteed that the stabilisers in
        the Tableau are exactly the same as the ones inputted (or whether they can be
        linear combinations of them, in which case the algorithm will think it is a
        logical operator). No example has been found where this was an issue so far, but
        this is to be extra safe. By default, None.

    Returns
    -------
    tuple[tuple[set[_PauliGate], ...], tuple[set[_PauliGate], ...]]
        The logical operators, provided as a tuple of all the X logical
        operators at index 0 and all the Z logical operators at index 1.
        The logical operators are ordered in anticommuting pairs, such
        that the ith X logical commutes with all X and Z logical operators,
        except for the ith Z logical operator, with which it anticommutes.
    """
    # compute the mapping from qubit to index in the pauli string
    qubit_to_pauli_index: dict[Qubit, int] = {}
    index = 0
    for stabiliser in stabilisers:
        for pauli in stabiliser.paulis:
            if (pauli is not None) and (
                (qubit := pauli.qubit) not in qubit_to_pauli_index
            ):
                qubit_to_pauli_index[qubit] = index
                index += 1

    # convert the stabilisers to paulistring format
    paulistrings = [
        pauli_gates_to_stim_pauli_string(stabiliser.paulis, qubit_to_pauli_index)
        for stabiliser in stabilisers
    ]

    pauli_index_to_qubit = {v: k for k, v in qubit_to_pauli_index.items()}

    # compute the logical operators as paulistrings
    str_operators = get_str_logical_operators_from_tableau(
        paulistrings, num_logical_qubits
    )

    x_str_operators, z_str_operators = (
        list(zip(*str_operators)) if len(str_operators) > 0 else ([], [])
    )

    # convert paulistrings to operator format
    x_operators = tuple(
        set(paulistring_to_operator(opr, pauli_index_to_qubit))
        for opr in x_str_operators
    )
    z_operators = tuple(
        set(paulistring_to_operator(opr, pauli_index_to_qubit))
        for opr in z_str_operators
    )

    return x_operators, z_operators


def get_logical_operators_from_css_parity_check_matrices(
    hx: NDArray, hz: NDArray, column_to_qubit: dict[int, Qubit]
) -> tuple[tuple[set[PauliX], ...], tuple[set[PauliZ], ...]]:
    """
    For a CSS stabiliser code, computes the logical operators using its parity check
    matrices and the BPOSD package (https://arxiv.org/abs/2005.07016).

    This method of computing the logical operators:
    - Does NOT guarantee the logical operators are independent,
    - Does NOT guarantee the logical operators are minimum-weight,
    - Guarantees for CSS codes that X logical operators are made purely of X gates
    and Z logical operators are made purely purely of Z gates (this is suspected but
    wasn't decidedly shown).

    Parameters
    ----------
    h_x : NDArray
        The check matrix (containing only 0 and 1) for X stabilisers where each
        row represents an X stabiliser. If an empty matrix, then this means the
        CSS code has no X stabilisers.
    h_z : NDArray
        The check matrix (containing only 0 and 1) for Z stabilisers where each
        row represents a Z stabiliser. If an empty matrix, then this means the
        CSS code has no Z stabilisers.
    column_to_qubit : dict[int, Qubit]
        A mapping from column index in the parity check matrices h_x, h_z to the
        data qubit the column's entries describe.

    Returns
    -------
    tuple[tuple[set[PauliX], ...], tuple[set[PauliZ], ...]]
        The logical operators, provided as a tuple of all the X logical operators
        at index 0 and all the Z logical operators at index 1.
    """
    code = css_code(hx, hz)
    x_logs, z_logs = code.compute_logicals()

    return tuple(
        set(PauliX(column_to_qubit[i]) for i, x in enumerate(log_op) if x)
        for log_op in x_logs
    ), tuple(
        set(PauliZ(column_to_qubit[i]) for i, x in enumerate(log_op) if x)
        for log_op in z_logs
    )
