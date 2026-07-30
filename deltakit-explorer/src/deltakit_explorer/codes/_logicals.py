# (c) Copyright Riverlane 2020-2025.
"""
This module defines a function to compute the logical operators associated
with a collection of stabilisers.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

import galois
import numpy as np
from deltakit_circuit import PauliX, PauliY, PauliZ, Qubit
from deltakit_circuit._qubit_identifiers import _PauliGate
from deltakit_stim import PauliString, Tableau
from numpy.typing import NDArray

from deltakit_explorer.codes._css._stabiliser_helper_functions import (
    pauli_gates_to_stim_pauli_string,
)
from deltakit_explorer.codes._stabiliser import Stabiliser


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
    stabilisers: Collection[PauliString], num_logical_qubits: int | None = None
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
    stabilisers: Iterable[Stabiliser], num_logical_qubits: int | None = None
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
    x_logs, z_logs = css_code_compute_logicals(hx, hz)

    return tuple(
        {PauliX(column_to_qubit[i]) for i, x in enumerate(log_op) if x}
        for log_op in x_logs
    ), tuple(
        {PauliZ(column_to_qubit[i]) for i, x in enumerate(log_op) if x}
        for log_op in z_logs
    )


def pivot_rows(parity_check_matrix: galois.FieldArray) -> NDArray[np.int_]:
    """
    Compute the pivot row indices of a dense binary check matrix over GF(2).

    Args:
        parity_check_matrix: A dense binary check matrix represented as a
            ``galois.FieldArray`` over GF(2).

    Returns:
        The pivot row indices of ``parity_check_matrix``.
    """
    # Pivot rows of ``parity_check_matrix`` are the non-zero pivot columns of
    # ``parity_check_matrix.T``.
    row_reduced = parity_check_matrix.T.row_reduce()
    pivots: list[int] = []
    for row in row_reduced:
        non_zero_row = np.flatnonzero(row != 0)
        if non_zero_row.size:
            pivots.append(int(non_zero_row[0]))
    return np.asarray(pivots, dtype=np.int_)


def css_code_compute_logicals(
    hx: NDArray[np.floating],
    hz: NDArray[np.floating],
    *,
    lx_preferred: NDArray[np.floating] | None = None,
    lz_preferred: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Drop-in replacement for calling bposd.css_code.compute_logicals.

    Note:
        This function has been taken from the following repository under the MIT
        licence: [bp_osd repository](https://github.com/quantumgizmos/bp_osd).

        The following modifications were performed:
            1. Add more detailed typing information (parameter types, return type).
            2. Remove two lines of code that were not directly used by the method:
                ```
                if self.K == np.nan:
                    self.compute_dimension()
                ```
            3. Change the names of the parameters of compute_lz to avoid a name clash.
            4. Remove all the `self.`.
            5. Add a docstring.
            6. Add typing to the internal `compute_lz` function.
            7. Only use dense matrices because the inputs are dense anyway.
            8. Add ``lx_preferred``/``lz_preferred`` to let a caller bias which
               representative is picked for a logical class, since the basis a
               plain kernel/quotient computation returns is not unique and a
               caller may rely on a structural property that isn't guaranteed
               by an arbitrary basis.

        You can check the original version of this function at
        [this permalink](https://github.com/quantumgizmos/bp_osd/blob/8894ec654b24ae875c07e5a361dcae9a77d748ce/src/bposd/css.py#L75).

    Args:
        hx: parity check matrix for the X code.
        hz: parity check matrix for the Z code.
        lx_preferred: optional binary vectors, each already lying in
            ``ker(hz)``, to prefer as representatives of their logical class.
            Vectors that are stabilisers or dependent on earlier rows are
            skipped and contribute nothing extra.
        lz_preferred: as ``lx_preferred``, but for Z logicals, each already
            lying in ``ker(hx)``.

    Returns:
        a tuple ``(lx, lz)`` representing the X and Z logicals.
    """

    def validate_preferred_logicals(
        _preferred: NDArray[np.floating],
        _hx_gf: galois.FieldArray,
    ) -> None:
        """Validate optional preferred logical representatives.

        Args:
            _preferred: Binary row vectors to prioritise.
            _hx_gf: Check matrix whose kernel must contain ``_preferred``.

        Raises:
            ValueError: If the preferred rows are non-binary or do not lie in
                the required kernel.
        """
        if not np.all((_preferred == 0) | (_preferred == 1)):
            msg = "Preferred logicals must be binary."
            raise ValueError(msg)
        if not np.all((_preferred @ np.asarray(_hx_gf.T, dtype=np.int_)) % 2 == 0):
            msg = "Preferred logicals must lie in the kernel of the check matrix."
            raise ValueError(msg)

    def compute_lz(
        _hx: NDArray[np.floating],
        _hz: NDArray[np.floating],
        _preferred: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """Compute a basis of logical operators for one side of a CSS code.

        This function finds operators that satisfy the checks in ``_hx`` but are
        not just combinations of the checks in ``_hz``.

        Args:
            _hx: Check matrix used to define valid operators.
            _hz: Check matrix used to remove redundant operators.
            _preferred: Optional vectors, already lying in ``ker(_hx)``, to
                prefer as representatives of their logical class over other
                representatives that would otherwise be found.

        Returns:
            A binary matrix whose rows are logical operators.
        """
        _hx_gf = galois.GF2(np.asarray(_hx, dtype=np.int_))
        _hz_gf = galois.GF2(np.asarray(_hz, dtype=np.int_))

        ker_hx_gf = _hx_gf.null_space()
        rank_hz_gf = np.linalg.matrix_rank(_hz_gf)

        stack = [_hz_gf]
        if _preferred is not None:
            validate_preferred_logicals(_preferred, _hx_gf)
            preferred_gf = galois.GF2(_preferred.astype(np.int_))
            stack.append(preferred_gf)

        stack.append(ker_hx_gf)
        log_stack = np.vstack(stack)
        pivots = pivot_rows(log_stack)[rank_hz_gf:]

        return np.asarray(log_stack[pivots])

    return (
        compute_lz(hz, hx, lx_preferred),
        compute_lz(hx, hz, lz_preferred),
    )
