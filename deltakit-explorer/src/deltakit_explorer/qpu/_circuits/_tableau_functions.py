# (c) Copyright Riverlane 2020-2025.
"""
This module consists of functions enabling circuit compilation
using the stim Tableau.
"""
# pylint: disable=too-many-lines,too-many-nested-blocks,too-many-branches,too-many-statements
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import mul

import numpy as np
import stim
from deltakit_circuit import Circuit, GateLayer, Layer, Qubit
from deltakit_circuit.gates import (
    CX,
    CXSWAP,
    CY,
    CZ,
    CZSWAP,
    ISWAP,
    ISWAP_DAG,
    MEASUREMENT_GATE_MAPPING,
    MPP,
    MRX,
    MRY,
    MRZ,
    MX,
    MY,
    MZ,
    ONE_QUBIT_GATE_MAPPING,
    RESET_GATE_MAPPING,
    RX,
    RY,
    RZ,
    S_DAG,
    SQRT_XX,
    SQRT_XX_DAG,
    SQRT_YY,
    SQRT_YY_DAG,
    SQRT_ZZ,
    SQRT_ZZ_DAG,
    TWO_QUBIT_GATE_MAPPING,
    XCX,
    XCY,
    XCZ,
    YCX,
    YCY,
    YCZ,
    Gate,
    H,
    I,
    S,
)

from deltakit_explorer.qpu._native_gate_set import (
    NativeGateSetAndTimes,
    OneQubitCliffordGate,
    TwoOperandGate,
)

# key: tuple of the form (gate's layer index, gate's qubit, gate's stim_string representation)
# values: dictionary containing preceding and succeeding unitary block indices:
#   {"preceding": 0, "succeeding": 1}
SpecialGateDict = dict[tuple[int, Qubit, str], dict[str, int]]

# key: tableau in str form: ("+X", "+Z")
# values: tuple of gates that give the tableau, as strs: ("X", "X")
TableauDict = dict[tuple[str, ...], tuple[str, ...]]

# key: tableau, in str form: ("+X", "+Z")
# values: list of possible unitary gates with that tableau: [[I], [X, X], [Z, Z], ...]
EquivalentTableauDict = dict[tuple[str, str], list[list[OneQubitCliffordGate]]]

# A single entry in the CompilationData.two_qubit_gates dictionary. The first
# item in the tuple describes the gate layer index, qubit, and gate.stim_string
# for the given gate. The second entry is the unitary block dictionary, with keys
# being "preceding" or "succeeding" and values being the unitary block index.
TwoQubitGateDictEntry = tuple[tuple[int, Qubit, str], dict[str, int]]

# Dictionary to look up compilation of a two qubit in terms of additional unitary
# gates required. The key can be either a gate currently being compiled or the native
# gate being compiled to. The values are the unitary blocks needed to do the compilation.
# The order is ub1,ub2,ub3,ub4, corresponding to before and after qubit1 and before and
# after qubit 2 respectively.
TwoQubitGateCompilationLookupDict = dict[
    TwoOperandGate,
    tuple[
        list[OneQubitCliffordGate],
        list[OneQubitCliffordGate],
        list[OneQubitCliffordGate],
        list[OneQubitCliffordGate],
    ],
]

MEAS_COMPILATION_LOOKUP_DICT: dict[
    MX | MY | MZ | MRX | MRY | MRZ,
    dict[
        MX | MY | MZ | MRX | MRY | MRZ,
        tuple[list[OneQubitCliffordGate], list[OneQubitCliffordGate]],
    ],
] = {
    MX: {MX: ([], []), MY: ([S], [S_DAG]), MZ: ([H], [H])},
    MY: {MX: ([S_DAG], [S]), MY: ([], []), MZ: ([S_DAG, H], [H, S])},
    MZ: {MX: ([H], [H]), MY: ([H, S], [S_DAG, H]), MZ: ([], [])},
}
# key is the gate you're compiling from, value is the gate
# you're compiling to
RESET_COMPILATION_LOOKUP_DICT: dict[
    RX | RY | RZ,
    dict[
        RX | RY | RZ,
        tuple[list[OneQubitCliffordGate], list[OneQubitCliffordGate]],
    ],
] = {
    RX: {
        RX: ([], MEAS_COMPILATION_LOOKUP_DICT[MX][MX][1]),
        RY: ([], MEAS_COMPILATION_LOOKUP_DICT[MX][MY][1]),
        RZ: ([], MEAS_COMPILATION_LOOKUP_DICT[MX][MZ][1]),
    },
    RY: {
        RX: ([], MEAS_COMPILATION_LOOKUP_DICT[MY][MX][1]),
        RY: ([], MEAS_COMPILATION_LOOKUP_DICT[MY][MY][1]),
        RZ: ([], MEAS_COMPILATION_LOOKUP_DICT[MY][MZ][1]),
    },
    RZ: {
        RX: ([], MEAS_COMPILATION_LOOKUP_DICT[MZ][MX][1]),
        RY: ([], MEAS_COMPILATION_LOOKUP_DICT[MZ][MY][1]),
        RZ: ([], MEAS_COMPILATION_LOOKUP_DICT[MZ][MZ][1]),
    },
}
MEAS_COMPILATION_LOOKUP_DICT[MRX] = {
    MRX: MEAS_COMPILATION_LOOKUP_DICT[MX][MX],
    MRY: MEAS_COMPILATION_LOOKUP_DICT[MX][MY],
    MRZ: MEAS_COMPILATION_LOOKUP_DICT[MX][MZ],
}
MEAS_COMPILATION_LOOKUP_DICT[MRY] = {
    MRX: MEAS_COMPILATION_LOOKUP_DICT[MY][MX],
    MRY: MEAS_COMPILATION_LOOKUP_DICT[MY][MY],
    MRZ: MEAS_COMPILATION_LOOKUP_DICT[MY][MZ],
}
MEAS_COMPILATION_LOOKUP_DICT[MRZ] = {
    MRX: MEAS_COMPILATION_LOOKUP_DICT[MZ][MX],
    MRY: MEAS_COMPILATION_LOOKUP_DICT[MZ][MY],
    MRZ: MEAS_COMPILATION_LOOKUP_DICT[MZ][MZ],
}
# Dictionary to go from the current (CPSWAP-like) gate to CZSWAP. The key
# is the current gate being compiled, and value is a tuple containing
# unitary blocks to do the compilation. The order is ub1,ub2,ub3,ub4,
# corresponding to before and after qubit1 and before and after qubit 2
# respectively. Therefore, it provides the following compilation (in circuit order):
# Gate(a, b) = ub1(a) ub3(b) CZSWAP(a, b) ub2(a) ub4(b)
GATE_TO_CZSWAP_DICT: TwoQubitGateCompilationLookupDict = {
    ISWAP: ([], [S], [], [S]),
    ISWAP_DAG: ([S_DAG], [], [S_DAG], []),
    CXSWAP: ([], [H], [H], []),
    CZSWAP: ([], [], [], []),
}
# Dictionary to go from the CZSWAP to the target gate. The key
# is the target gate, and value is a tuple containing
# unitary blocks to do the compilation. The order is ub1,ub2,ub3,ub4,
# corresponding to before and after qubit1 and before and after qubit 2
# respectively. Therefore, it provides the following compilation (in circuit order):
# CZSWAP(a, b) = ub1(a) ub3(b) Gate(a, b) ub2(a) ub4(b)
CZSWAP_TO_GATE_DICT: TwoQubitGateCompilationLookupDict = {
    ISWAP: ([], [S_DAG], [], [S_DAG]),
    ISWAP_DAG: ([S], [], [S], []),
    CXSWAP: ([], [H], [H], []),
    CZSWAP: ([], [], [], []),
}
# Dictionary to go from the current (CP-like) gate to CZ. The key
# is the current gate being compiled, and value is a tuple containing
# unitary blocks to do the compilation. The order is ub1,ub2,ub3,ub4,
# corresponding to before and after qubit1 and before and after qubit 2
# respectively. Therefore, it provides the following compilation (in circuit order):
# Gate(a, b) = ub1(a) ub3(b) CZ(a, b) ub2(a) ub4(b)
GATE_TO_CZ_DICT: TwoQubitGateCompilationLookupDict = {
    # qubit0_bef, qubit0_aft, qubit1_bef, qubit1_aft
    CX: ([], [], [H], [H]),
    CY: ([], [], [S_DAG, H], [H, S]),
    CZ: ([], [], [], []),
    SQRT_XX: ([H], [S, H], [H], [S, H]),
    SQRT_XX_DAG: ([H], [S_DAG, H], [H], [S_DAG, H]),
    SQRT_YY: ([S_DAG, H], [S, H, S], [S_DAG, H], [S, H, S]),
    SQRT_YY_DAG: ([S_DAG, H], [S, H, S], [S, H], [S, H, S_DAG]),
    SQRT_ZZ: ([], [S], [], [S]),
    SQRT_ZZ_DAG: ([], [S_DAG], [], [S_DAG]),
    XCX: ([H], [H], [H], [H]),
    XCY: ([H], [H], [S_DAG, H], [H, S]),
    XCZ: ([H], [H], [], []),
    YCX: ([S_DAG, H], [H, S], [H], [H]),
    YCY: ([S_DAG, H], [H, S], [S_DAG, H], [H, S]),
    YCZ: ([S_DAG, H], [H, S], [], []),
}
# Dictionary to go from the CZ to the target gate. The key
# is the target gate, and value is a tuple containing
# unitary blocks to do the compilation. The order is ub1,ub2,ub3,ub4,
# corresponding to before and after qubit1 and before and after qubit 2
# respectively. Therefore, it provides the following compilation (in circuit order):
# CZ(a, b) = ub1(a) ub3(b) Gate(a, b) ub2(a) ub4(b)
CZ_TO_GATE_DICT: TwoQubitGateCompilationLookupDict = {
    CX: ([], [], [H], [H]),
    CY: ([], [], [H, S], [S_DAG, H]),
    CZ: ([], [], [], []),
    SQRT_XX: ([H], [H, S_DAG], [H], [H, S_DAG]),
    SQRT_XX_DAG: (
        [H],
        [H, S],
        [H],
        [H, S],
    ),
    SQRT_YY: (
        [H, S],
        [S_DAG, H, S_DAG],
        [H, S],
        [S_DAG, H, S_DAG],
    ),
    SQRT_YY_DAG: (
        [H, S],
        [S_DAG, H, S_DAG],
        [H, S_DAG],
        [S, H, S_DAG],
    ),
    SQRT_ZZ: ([], [S_DAG], [], [S_DAG]),
    SQRT_ZZ_DAG: (
        [],
        [S],
        [],
        [S],
    ),
    XCX: ([H], [H], [H], [H]),
    XCY: ([H], [H], [H, S], [S_DAG, H]),
    XCZ: ([H], [H], [], []),
    YCX: ([H, S], [S_DAG, H], [H], [H]),
    YCY: (
        [H, S],
        [S_DAG, H],
        [H, S],
        [S_DAG, H],
    ),
    YCZ: ([H, S], [S_DAG, H], [], []),
}


def _get_tableau_from_sequence_of_gates(
    unitary_block: Sequence[OneQubitCliffordGate],
) -> stim.Tableau:
    return reduce(
        mul,
        (
            stim.Tableau.from_named_gate(gate.stim_string)
            for gate in unitary_block[::-1]
            # ensuring circuit order, since lestim assumes matrix order for mul
        ),
        stim.Tableau.from_named_gate("I"),  # default in case unitary_block empty
    )


def _get_tableau_as_key(
    tableau: stim.Tableau, up_to_paulis: bool = False
) -> tuple[str, str]:
    """
    Given a stim.Tableau, corresponding to a one-qubit
    Clifford unitary, return a tuple of the string output
    of the Tableau to create a hashable key. Uses the
    stim.Tableau.x_output, stim.Tableau.z_output, in that order
    to produce the output.
    Returns the output with or without sign information, as per
    up_to_paulis.
    E.g, ("+X", "+Z") if up_to_paulis=False, or ("X", "Z") otherwise.

    Parameters
    ----------
    tableau : stim.Tableau
        The tableau to be turned into a hashable key.
    up_to_paulis : bool, optional
        Specify whether signs should be included in the key.

    Returns
    -------
    Tuple[str, str]
        A tuple of the string values of the tableau output.
        E.g, ("+X", "+Z"), or ("X", "Z") if up_to_paulis=True.
    """
    return (
        str(tableau.x_output(0))[int(up_to_paulis) :],
        str(tableau.z_output(0))[int(up_to_paulis) :],
    )


def _get_tableau_from_sequence_of_1q_gates(gates: Sequence[str]) -> stim.Tableau:
    if len(gates) == 0:
        return stim.Tableau.from_named_gate("I")
    try:
        return reduce(
            mul,
            (
                stim.Tableau.from_named_gate(gate)
                for gate in gates[::-1]
                # ensuring circuit order, since lestim assumes matrix order for mul
            ),
        )
    except ValueError as ve:
        msg = "'gates' must be composed of only single qubit gates"
        raise ValueError(msg) from ve


def _get_compilation_dict(
    native_gates: NativeGateSetAndTimes,
    max_length: int | None = None,
    up_to_paulis: bool = False,
) -> tuple[TableauDict, EquivalentTableauDict]:
    """
    Create a compilation dictionary that specifies, for a given set of single
    qubit native gates, the lowest length operator-product per tableau. To achieve this,
    starting from the identity, multiply in all the native single-qubit gates to create
    length one operators and add the unique tableaus to a dictionary, with tableau as
    key and corresponding operator as value. Then progressively multiply in the single-
    qubit native gates creating length 2, 3 etc. operators, keeping only the minimum-length
    ones for a unique set of tableaus, up to the maximum of 24, or until no new tableaus
    are found for a given operator length.

    Parameters
    ----------
    native_gates : NativeGateSetAndTimes
        Native gate set. If `one_qubit_gates` is empty, return
        a dictionary only containing the identity.
    max_length : Optional[int], optional
        Maximum length of operators to be considered, used to put an upper
        limit on which combinations of the single-qubit native gates should
        be created.
        By default, None, which will let the program run until it either
        finds all tableaus or determines there are none left to find.
    up_to_paulis : bool, optional
        If True, then when creating the Tableau dictionary, only consider
        Tableaus that are equivalent up to Pauli terms. This means not
        considering the sign of the conjugation result. The maximum number
        of unique Tableaus then becomes 6.
        By default, False.

    Returns
    -------
    Tuple[TableauDict, EquivalentTableauDict]
        A tuple containing the compilation dictionary, and the equivalent
        tableau dictionary.

        Compilation dictionary: TableauDict
            A dictionary with Tableaus as keys and operators as values.
            E.g, `{("+X", "-Z"): ("X",), ("+X", "+Z"): ()}`
            The tableau is created from stim.Tableau.x_output and
            stim.Tableau.z_output, in that order.
            The value is the empty tuple if the tableau key is equivalent to the identity.
        Equivalent tableau dictionary: EquivalentTableauDict
            A dictionary with Tableaus as keys and lists of lists of OneQubitCliffordGate
            as values, where the list expresses the gates in circuit order.
            E.g, `{("+Z", "-X"): [[H, Z], [X, H]], ...}`
    """

    if up_to_paulis:
        max_unique_tableau_count = 6
        identity_tableau = ("X", "Z")
        one_q_gates_as_stim_string = {
            str(g.stim_string)
            for g in native_gates.one_qubit_gates
            if str(g.stim_string) not in ["X", "Y", "Z"]
        }
    else:
        max_unique_tableau_count = 24
        identity_tableau = ("+X", "+Z")
        one_q_gates_as_stim_string = {
            str(g.stim_string) for g in native_gates.one_qubit_gates
        }

    # Include the identity tableau, as it is always there by default
    min_weight_tableau_dict: TableauDict = {identity_tableau: ()}
    equiv_tableau_dict: defaultdict[
        tuple[str, str], list[list[OneQubitCliffordGate]]
    ] = defaultdict(list)
    current_weight = 1
    while max_length is None or current_weight <= max_length:
        # keep track of whether we've inserted any new tableaus. if we haven't,
        # we can stop the computation there
        new_tableau_found = False
        # if True, return dicts at end of current gate product iteration weight.
        # e.g, if set True on weight 3 products, will iterate through all the remaining
        # weight 3 products before returning. this is to catch equivalent tableaus remaining
        finish_at_end_of_product_loop = False

        # need to make a copy since we will change the dictionary during the loop
        copy_of_dict_to_iterate_over = min_weight_tableau_dict.copy()
        for current_gates in copy_of_dict_to_iterate_over.values():
            if len(current_gates) != current_weight - 1:
                continue
            for native_gate in one_q_gates_as_stim_string:
                gate_product = (*current_gates, native_gate)
                product_tableau = _get_tableau_from_sequence_of_1q_gates(gate_product)
                tableau_key = _get_tableau_as_key(product_tableau, up_to_paulis)

                # if new tableau found, add it, it will be minimum weight
                if tableau_key not in min_weight_tableau_dict:
                    min_weight_tableau_dict[tableau_key] = gate_product
                    equiv_tableau_dict[tableau_key].append(
                        [ONE_QUBIT_GATE_MAPPING[g] for g in gate_product]
                    )
                    new_tableau_found = True
                    if len(min_weight_tableau_dict) == max_unique_tableau_count:
                        # we've found all possible tableaus
                        # dont return too soon, as we want to find equivalent tableaus
                        finish_at_end_of_product_loop = True

                # if tableau already in dict, consider it for equivalent tableau dict
                # only add equivalent gate expressions of minimum length
                elif len(gate_product) <= len(min_weight_tableau_dict[tableau_key]):
                    equiv_tableau_dict[tableau_key].append(
                        [ONE_QUBIT_GATE_MAPPING[g] for g in gate_product]
                    )

        if not new_tableau_found or finish_at_end_of_product_loop:
            # since we didn't find any new tableaus, we can stop
            return min_weight_tableau_dict, equiv_tableau_dict
        current_weight += 1
    return min_weight_tableau_dict, equiv_tableau_dict


def _get_tableau_z_image_as_string(tableau: stim.Tableau, up_to_paulis: bool):
    return str(tableau.z_output(0))[int(up_to_paulis) :]


def _get_tableau_x_image_as_string(tableau: stim.Tableau, up_to_paulis: bool):
    return str(tableau.x_output(0))[int(up_to_paulis) :]


def _get_tableau_y_image_as_string(tableau: stim.Tableau, up_to_paulis: bool):
    return str(tableau.y_output(0))[int(up_to_paulis) :]


def _get_compilation_with_projectors_before_unitaries(
    compilation_dict: TableauDict,
    unitary_block: Sequence[OneQubitCliffordGate],
    projector_before_unitaries: MZ | RZ | MX | RX | MY | RY,
    up_to_paulis: bool = False,
) -> tuple[str, ...]:
    """
    Given a string/block of unitary gates, and specifying which projector
    this unitary block is preceded by (reset or measurement), use the Tableau
    to work out if this set of projector + unitaries can be reduced to a
    shorter set of gates. This requires comparing Tableaus that are identical in the
    conjugation corresponding to the basis of the projector. E.g, if two Tableaus
    conjugate +Z to +X, then the Tableau with the shortest gate representation may
    be chosen to follow a reset or measurement in the Z basis. Likewise for the other
    bases. If the given set of unitaries is empty, the Identity will be returned.
    If the compilation dictionary contains no tableau that matches that of the
    `unitary_block`, then a KeyError is thrown, as this may cause a return value in
    terms of non-native gates.
    If the compilation is requested in the Y basis, e.g, MY or RY is provided, the
    Y tableau of the dictionary components will be computed from their X and Z parts.

    Parameters
    ----------
    compilation_dict : TableauDict
        Tableau dict calculated for the native gate set corresponding to
        the circuit being compiled. Values (gates) follow circuit order.
    unitary_block : Sequence[OneQubitCliffordGate]
        Unitary gates to try to reduce by comparing Tableaus, specified
        in circuit order.
    projector_before_unitaries : Union[MZ, RZ, MX, RX, MY, RY]
        Specify which projector is immediately before
        the unitary_block in the circuit.
    up_to_paulis : bool, Optional
        If True, the tableau computed for the unitary block
        will be computed only up to Paulis, meaning the sign information
        is not considered in comparisons with `compilation_dict`.
        The provided `compilation_dict` should match with the choice
        of up_to_paulis.
        False by default.

    Returns
    -------
    Tuple[str, ...]
        Shortest length gate series corresponding to identical action under
        Tableau for the relevant basis of projector_before_unitaries.
    """
    # first, get the Tableau for the unitary_block
    if len(unitary_block) == 0:
        unitary_block_tableau: stim.Tableau = stim.Tableau.from_named_gate("I")
        shortest_gates: tuple = ()
    else:
        unitary_block_tableau = _get_tableau_from_sequence_of_1q_gates(
            [gate.stim_string for gate in unitary_block]
        )
        shortest_gates = tuple(str(g.stim_string) for g in unitary_block)

    # use appropriate function to get tableau image depending on basis
    if projector_before_unitaries in [MZ, RZ]:
        get_tableau_image_as_string = _get_tableau_z_image_as_string
        tableau_index = 1
    elif projector_before_unitaries in [MX, RX]:
        get_tableau_image_as_string = _get_tableau_x_image_as_string
        tableau_index = 0
    elif projector_before_unitaries in [MY, RY]:
        get_tableau_image_as_string = _get_tableau_y_image_as_string
        # update tableau to include Y term at index 2
        tableau_index = 2
        compilation_dict_with_y = compilation_dict.copy()
        for tableau, gates in list(compilation_dict.items()):
            compilation_dict_with_y[
                (
                    *tableau,
                    str(
                        stim.Tableau.from_conjugated_generators(
                            xs=[stim.PauliString(tableau[0])],
                            zs=[stim.PauliString(tableau[1])],
                        ).y_output(0)
                    )[
                        1 if up_to_paulis else 0 :
                    ],  # remove sign if not needed
                )
            ] = gates
            del compilation_dict_with_y[tableau]
        compilation_dict = compilation_dict_with_y
    else:
        msg = f"{projector_before_unitaries.stim_string} is not a recognised projector"
        raise NotImplementedError(msg)

    # for the basis of the reset or measurement before a unitary block, we can just consider
    # the part of the Tableau relevant to the basis.
    unitary_block_tableau_in_comp_dict = False
    for tableau, gates in compilation_dict.items():
        if tableau[tableau_index] == get_tableau_image_as_string(
            unitary_block_tableau, up_to_paulis
        ):
            unitary_block_tableau_in_comp_dict = True
            if len(gates) < len(shortest_gates):
                shortest_gates = gates

    if not unitary_block_tableau_in_comp_dict:
        msg = (
            "unitary_block's tableau is not in the compilation_dictionary."
            " This means the output of this function may include gates not"
            " in the native gate set. Try compiling unitary_block to the"
            " native gate set first."
        )
        raise KeyError(msg)

    return shortest_gates


def _get_compilation_with_measurement_after_unitaries(
    compilation_dict: TableauDict,
    unitary_block: Sequence[OneQubitCliffordGate],
    measurement_after_unitaries: MZ | RZ | MX | RX | MY | RY,
    up_to_paulis: bool = False,
) -> tuple[str, ...]:
    """
    In the case of a measurement following a block of unitary operators,
    we may simply consider the Tableau's that treat the measurement basis
    Pauli in an equivalent way. To this end, we consider the Tableau of
    the unitary block, and see under which conjugation the measurement
    basis is maintained. Then, we search the compilation dictionary for
    equivalent Tableaus that offer the same conjugation. This is achieved
    by calling `_get_compilation_with_projectors` with `projector_before_unitaries`
    chosen to be the conjugation that gives the unchanged measurement basis.

    Parameters
    ----------
    compilation_dict : TableauDict
        Tableau dict calculated for the native gate set corresponding to
        the circuit being compiled.
    unitary_block : Collection[OneQubitCliffordGate]
        Unitary gates to try to reduce by comparing Tableaus.
    measurement_after_unitaries : Union[MZ, MX, MY]
        Specify which measurement is immediately after
        the unitary_block in the circuit.
    up_to_paulis : bool, Optional
        If True, the tableau computed for the unitary block
        will be computed only up to Paulis, meaning the sign information
        is not considered in comparisons with `compilation_dict`.
        The provided `compilation_dict` should match with the choice
        of up_to_paulis.
        False by default.

    Returns
    -------
    Tuple[str, ...]
        Shortest length gate series corresponding to identical action under
        Tableau for the relevant basis of measurement_after_unitaries.
    """
    unitary_block_tableau = _get_tableau_from_sequence_of_gates(unitary_block)
    if (
        str(unitary_block_tableau.x_output(0))[1:]
        == measurement_after_unitaries.basis.value
    ):
        new_basis = RX
    elif (
        str(unitary_block_tableau.z_output(0))[1:]
        == measurement_after_unitaries.basis.value
    ):
        new_basis = RZ
    else:
        new_basis = RY

    return _get_compilation_with_projectors_before_unitaries(
        compilation_dict, unitary_block, new_basis, up_to_paulis
    )


def _is_identity_like(tableau: stim.Tableau) -> bool:
    """
    Function to test whether a tableau is identity-like. This is achieved via
    comparing to the identity Tableau in the numpy form of the tableau.
    If a tableau is identity like, the X-to-X and Z-to-Z components will be
    the same as the identity matrix, and the X-to-Z and Z-to-X components will
    be equal to the all-zero matrix. If this is not the case, we deduce that
    the tableau is not identity-like.

    Parameters
    ----------
    tableau : stim.Tableau
        Tableau to compare to identity.

    Returns
    -------
    bool
        True if the tableau is identity-like, false otherwise.
    """
    x2x, x2z, z2x, z2z, _, _ = tableau.to_numpy()
    for identity_component in [x2x, z2z, x2z, z2x]:
        # check if any off-diagonal elements are non-zero
        if (identity_component[~np.eye(len(identity_component), dtype=np.bool_)]).any():
            return False
    return True


def _get_tableau_key_from_sequence_of_gates(
    gates: Sequence[OneQubitCliffordGate], up_to_paulis: bool = False
) -> tuple[str, str]:
    return _get_tableau_as_key(
        _get_tableau_from_sequence_of_1q_gates([gate.stim_string for gate in gates]),
        up_to_paulis,
    )


def _get_single_qubits_tableau_key_from_two_qubit_tableau(
    two_qubit_tableau: stim.Tableau, qubit_index: int, up_to_paulis: bool = False
) -> tuple[str, str]:
    """
    Given a tableau for a two-qubit gate, get the part of the tableau that corresponds
    to just one of the qubits, specified by qubit_index.
    May also specify up_to_paulis to get the key without signs.

    Parameters
    ----------
    two_qubit_tableau : stim.Tableau
        Tableau for the two qubit gate.
    qubit_index : int
        Index of the single qubit portion of the tableau to return.
    up_to_paulis : bool, optional
        Specify whether signs should be included in the key or not.

    Returns
    -------
    Tuple[str, str]
        Tableau key for the qubit_index part of two_qubit_tableau.
        E.g, ("+Z", "+X") or ("Z", "X") if up_to_paulis=True.
    """
    if two_qubit_tableau.to_unitary_matrix(endian="little").shape != (4, 4):
        msg = "The given tableau does not describe a two qubit gate"
        raise ValueError(msg)
    if qubit_index == 0:
        tableau_key = (
            str(two_qubit_tableau.x_output(qubit_index))[
                :-1  # looks like "+Z_", so remove underscore here
            ][
                int(up_to_paulis) :
            ],  # remove +/- if up_to_paulis True
            str(two_qubit_tableau.z_output(qubit_index))[:-1][int(up_to_paulis) :],
        )
    else:
        tableau_key = (
            str(two_qubit_tableau.x_output(1))[
                0:3:2  # looks like "+_Z", so remove underscore here
            ][
                int(up_to_paulis) :
            ],  # remove +/- if up_to_paulis True
            str(two_qubit_tableau.z_output(1))[0:3:2][int(up_to_paulis) :],
        )
    return tableau_key


def _get_compilation_with_two_qubit_gates(
    two_qubit_gate: TwoOperandGate,
    compilation_dict: TableauDict,
    unitary_block_1: list[OneQubitCliffordGate],
    unitary_block_2: list[OneQubitCliffordGate],
    unitary_block_3: list[OneQubitCliffordGate],
    unitary_block_4: list[OneQubitCliffordGate],
    gate_exchange_dict: EquivalentTableauDict | None = None,
    up_to_paulis: bool = False,
    allow_terms_to_mutate: bool = True,
    allow_terms_to_multiply: bool = True,
) -> tuple[
    list[OneQubitCliffordGate],
    list[OneQubitCliffordGate],
    list[OneQubitCliffordGate],
    list[OneQubitCliffordGate],
]:
    """
    Given the following circuit diagram:
                          ___
    control: ---| U1 |---|   |---| U2 |---
                         |   |
                         | G |
                         |   |
    target:  ---| U3 |---|___|---| U4 |---

    Compile a two-qubit gate G and its surrounding one-qubit unitary blocks
    U1, U2, U3, U4 to reduce the number of unitaries before the gate,
    at the cost of increasing the number of unitaries after the gate,
    which may allow further compilation of the unitaries after the
    gate to reduce the overall unitary gate count.
    U1, U2, U3, U4 are in chronological order.

    Parameters
    ----------
    two_qubit_gate : TwoOperandGate
        The two-qubit gate inbetween the unitary blocks.
    compilation_dict : TableauDict
        The compilation dictionary for this circuit. Used to reduce unitary
        blocks to possibly shorter sequences.
    unitary_block_1 : List[OneQubitCliffordGate]
        The unitary block in the top left of the diagram, before the first qubit.
    unitary_block_2 : List[OneQubitCliffordGate]
        The unitary block in the top right of the diagram, after the first qubit.
    unitary_block_3 : List[OneQubitCliffordGate]
        The unitary block in the bottom left of the diagram, before the second qubit.
    unitary_block_4 : List[OneQubitCliffordGate]
        The unitary block in the bottom right of the diagram, after the second qubit.
    gate_exchange_dict : EquivalentTableauDict, optional
        Dictionary to provide extra possible gate combinations for a given Tableau.
        Used to exchange unitary blocks for equivalent Tableaus to check if there is
        an alternative set of gates that may allow some further simplification.
        By default, empty dictionary.
    up_to_paulis : bool, optional
        If True, will consider tableaus equivalent if they are equivalent up
        to pauli terms. E.g, the sign of the tableau terms are not considered.
    allow_terms_to_mutate : bool, optional
        If True, will allow the case of pulling a single-qubit Pauli term through a
        two-qubit gate such that the resulting (possibly two-qubit) Pauli has a different
        term on either qubit.
        E.g, pulling an X through a CZ from the first qubit mutates into a Z term on the second qubit.
        By default, True.
    allow_terms_to_multiply : bool, optional = True
        If True, will allow the case of pulling a term through a two-qubit gate
        such that a second term on the other qubit is created. E.g, pulling an X through
        a CX from the first qubit mutates into an X on the first qubit and an X on the
        second qubit.
        By default, True.

    Returns
    -------
    Tuple[
        List[OneQubitCliffordGate],
        List[OneQubitCliffordGate],
        List[OneQubitCliffordGate],
        List[OneQubitCliffordGate],
    ]:
        Tuple containing the updated unitary blocks 1-4 in order.
    """
    if gate_exchange_dict is None:
        gate_exchange_dict = {}

    # get tableau and inverse tableau of two_qubit_gate
    two_qubit_tableau = stim.Tableau.from_named_gate(two_qubit_gate.stim_string)  # G
    two_qubit_tableau_inv = two_qubit_tableau.inverse()  # G_adj

    # iterate over these to reduce code duplication.
    # first or second qubit changes which parts we look at in some places
    for unitary_block_whole, is_second_qubit in zip(
        [unitary_block_1, unitary_block_3], [False, True]
    ):
        # track how many gates have been pulled through, to adjust what we look at
        # e.g, if we took 2 gates off the end for compilation, this increases by 2,
        # to stop us looking at those 2 gates again.
        offset = 0

        # we first look at the rightmost gate in the unitary block to see if it can
        # be pulled through. If not, we look at the 2 rightmost gates, etc until
        # we have considered unitary block as a whole.
        #  ...--A1-A2-A3--  |  outer_start_index
        #             ^        = len-1
        #          ^  ^        = len-2
        #       ^  ^  ^        = len-3
        for outer_start_index in range(len(unitary_block_whole) - 1, -1, -1):
            # within each sub-block of gates in unitary_block, see if there
            # is an equivalent gate expression that lets us pull something through
            unitary_block_equiv_gates: list[
                list[OneQubitCliffordGate]
            ] = gate_exchange_dict.get(
                _get_tableau_key_from_sequence_of_gates(
                    unitary_block_whole[
                        outer_start_index : len(unitary_block_whole) - offset
                    ],
                    up_to_paulis,
                ),
                [
                    unitary_block_whole[
                        outer_start_index : len(unitary_block_whole) - offset
                    ]
                ],
            )

            # e.g, for unitary_block in [[SQRT_X, S, SQRT_X], [S, SQRT_X, S], ...]
            for current_unitary_block_equiv in unitary_block_equiv_gates:
                # for each of the equivalent expressions of unitary_block we examine,
                # we should also iterate over it starting from the right, the same as the
                # outer loop, to catch cases where only a part of the equivalent expression
                # can be pulled through
                for inner_start_index in range(
                    len(current_unitary_block_equiv) - 1, -1, -1
                ):
                    # we look at the portion of unitary_block that starts from inner_start_index onwards
                    partial_unitary_block = current_unitary_block_equiv[
                        inner_start_index:
                    ]

                    # get tableau for unitary block A, and its two qubit version AI or IA,
                    # by adding it to the identity, in the right order dependent on if
                    # unitary_block is the first or second qubit
                    unitary_block_tab = _get_tableau_from_sequence_of_1q_gates(
                        [gate.stim_string for gate in partial_unitary_block]
                    )
                    two_qubit_unitary_block_tab = (
                        stim.Tableau.from_named_gate("I") + unitary_block_tab
                        if is_second_qubit
                        else unitary_block_tab + stim.Tableau.from_named_gate("I")
                    )

                    # get tableau of G_adj A G, for G our two qubit gate
                    # if this is not identity-like, stop
                    conj_a_with_2q_gate_and_inv = (
                        two_qubit_tableau
                        * two_qubit_unitary_block_tab
                        * two_qubit_tableau_inv
                    )

                    # we can only pull things through if the conjugated tableau is identity-like
                    if not _is_identity_like(conj_a_with_2q_gate_and_inv):
                        continue
                    # check for the two optional conditions
                    if not allow_terms_to_mutate:
                        two_qubit_unitary_block_tab_reverse = (
                            unitary_block_tab + stim.Tableau.from_named_gate("I")
                            if is_second_qubit
                            else stim.Tableau.from_named_gate("I") + unitary_block_tab
                        )
                        if (
                            # check that pulling u through G, u-G becomes only
                            # G-u, with u possibly being on either qubit or both, as long as
                            # the term has not changed. That is, a Z pulled through must remain
                            # as a Z, and not change to any other Pauli term.
                            conj_a_with_2q_gate_and_inv
                            not in (
                                two_qubit_unitary_block_tab,
                                unitary_block_tab + unitary_block_tab,
                                two_qubit_unitary_block_tab_reverse,
                            )
                        ):
                            continue
                    if not allow_terms_to_multiply:
                        identity_tab = ("X", "Z") if up_to_paulis else ("+X", "+Z")
                        qubit_to_check_for_created_term = 0 if is_second_qubit else 1
                        if (
                            _get_single_qubits_tableau_key_from_two_qubit_tableau(
                                conj_a_with_2q_gate_and_inv,
                                qubit_to_check_for_created_term,
                                up_to_paulis,
                            )
                            != identity_tab
                        ):
                            continue

                    # from here, our goal is to calculate W2 and W4 where A G U2 U4 = G W2 W4 is compiled
                    # optimally. We know this is possible as the identity-likeness test passed.

                    # multiply this tableau with u2+u4 and then read off the u2 and u4 parts individually
                    u2_tab = _get_tableau_from_sequence_of_1q_gates(
                        [gate.stim_string for gate in unitary_block_2]
                    )
                    u4_tab = _get_tableau_from_sequence_of_1q_gates(
                        [gate.stim_string for gate in unitary_block_4]
                    )
                    # W2 W4
                    ub_and_2q_tab_with_u2u4 = (
                        u2_tab + u4_tab
                    ) * conj_a_with_2q_gate_and_inv  # matrix order

                    # read conjugation for first and second qubits separately
                    # W2
                    first_qubit_tab = (
                        _get_single_qubits_tableau_key_from_two_qubit_tableau(
                            ub_and_2q_tab_with_u2u4, 0, up_to_paulis
                        )
                    )

                    # W4
                    second_qubit_tab = (
                        _get_single_qubits_tableau_key_from_two_qubit_tableau(
                            ub_and_2q_tab_with_u2u4, 1, up_to_paulis
                        )
                    )

                    # lookup this tableau in the compilation dict to find the relevant gate
                    try:
                        # W2
                        unitary_block_2 = [
                            ONE_QUBIT_GATE_MAPPING[
                                gate
                            ]  # we have the gate string, so change to actual gate class
                            for gate in compilation_dict[first_qubit_tab]
                        ]
                        # W4
                        unitary_block_4 = [
                            ONE_QUBIT_GATE_MAPPING[gate]
                            for gate in compilation_dict[second_qubit_tab]
                        ]
                    except KeyError as ke:
                        msg = f"Tableau missing from compilation dictionary: {ke!s}"
                        raise KeyError(msg) from ke

                    # update pre-2q gate unitaries
                    # = original block up to where our outer index is
                    #   + current version of block up to pulled through gates
                    offset += len(current_unitary_block_equiv) - inner_start_index
                    if is_second_qubit:
                        unitary_block_3 = (
                            unitary_block_3[:outer_start_index]
                            + current_unitary_block_equiv[:inner_start_index]
                        )
                    else:
                        unitary_block_1 = (
                            unitary_block_1[:outer_start_index]
                            + current_unitary_block_equiv[:inner_start_index]
                        )

                    # since we've compiled, we should start again from the new unitary blocks
                    return _get_compilation_with_two_qubit_gates(
                        two_qubit_gate,
                        compilation_dict,
                        unitary_block_1,
                        unitary_block_2,
                        unitary_block_3,
                        unitary_block_4,
                        gate_exchange_dict,
                        up_to_paulis,
                        allow_terms_to_mutate,
                        allow_terms_to_multiply,
                    )

    return (unitary_block_1, unitary_block_2, unitary_block_3, unitary_block_4)


@dataclass
class CompilationData:
    """
    Dataclass to represent the data extracted from a `deltakit.circuit.Circuit`,
    to be used in Tableau compilation of that circuit.

    Parameters
    ----------
    unitary_blocks : Dict[int, List[OneQubitCliffordGate]]
        All the one-qubit unitary blocks in the Circuit, indexed by integers.
        A unitary block is a sequence of one-qubit unitary gates preceding or succeeding
        a reset, measurement or two-qubit gate.
    reset_gates : SpecialGateDict
        A dictionary of all the reset gates in the circuit. Keys are tuples
        of the form (GateLayer index, qubit, gate type) with values being dictionaries
        of the form {"preceding": unitary_block_index, "succeeding": unitary_block_index}
    measurement_gates : SpecialGateDict
        A dictionary of all the measurement gates in the circuit. Keys are tuples
        of the form (GateLayer index, qubit, gate type) with values being dictionaries
        of the form {"preceding": unitary_block_index, "succeeding": unitary_block_index}
    two_qubit_gates : SpecialGateDict
        A dictionary of all the two-qubit gates in the circuit. Keys are tuples
        of the form (GateLayer index, qubit, gate type) with values being dictionaries
        of the form {"preceding": unitary_block_index, "succeeding": unitary_block_index}
    non_gatelayer_layers : Dict[int, Layer]
        A dictionary of the layer index of a non-GateLayer layer. Keys are the layer's index,
        the values are the layer itself.
    iterations : int
        Number of iterations of the circuit, to represent repeat blocks.
    num_layers : int
        Number of layers in the circuit
    """

    unitary_blocks: dict[int, list[OneQubitCliffordGate]]
    reset_gates: SpecialGateDict
    measurement_gates: SpecialGateDict
    two_qubit_gates: SpecialGateDict
    non_gatelayer_layers: dict[int, Layer]
    iterations: int = 1
    num_layers: int = 0


def _get_relevant_dict_to_update(
    gate: Gate,
    reset_dict: SpecialGateDict,
    meas_dict: SpecialGateDict,
    two_q_dict: SpecialGateDict,
) -> SpecialGateDict:
    if isinstance(gate, (RX, RY, RZ)):
        return reset_dict
    if isinstance(gate, (MRX, MRY, MRZ, MX, MY, MZ)):
        return meas_dict
    if isinstance(gate, TwoOperandGate):
        return two_q_dict
    msg = f"Gate is not MR, M, R or 2Q: {gate}"
    raise ValueError(msg)


def _extract_structure_from_circuit(
    circuit: Circuit,
) -> CompilationData:
    """
    From the given circuit, extract the data relevant for compiling the circuit using
    the Tableau. This means we need to have collections of one-qubit unitary blocks
    that precede or succeed reset, measurement and two-qubit gates. In order to
    simplify compilation and reduce implementation complexity, each collection of
    reset, measurement and two-qubit gates will point to a shared memory of unitary
    blocks. This way, when one set of unitary blocks is updated after compilation of,
    for example, resets, the result will automatically propagate forward for when the
    same block is reconsidered but as a part of a different gate, say, a measurement
    later on in the circuit. This removes the complexity of having to keep track of
    which gates have compiled at what time, for which gate etc.

    The data structures are as follows:
        unitary blocks: returned as first item in the tuple.
            Stored as a dictionary of int keys and List[OneQubitCliffordGate] values.
            The key is the index of the unitary block that is pointed to by the reset, measurement
            and two-qubit gate dictionaries. The list is the unitary block itself.
        SpecialGateDict
            This data structure stores the relevant data for the `special` gates, those being
            reset, measurement and two-qubit gates. This dictionary has Tuple[int, Qubit] keys,
            and Dict["preceding": int, "succeeding": int] values.
            The key is (GateLayer index of gate, Qubit of gate, Gate). The value is two integers keyed
            by strings `preceding` and `succeeding` that point to a unitary block via that block's
            index.

        For example, if there is the small circuit
        ---RZ(0)--H(0)--MZ(0)---
        Then there are 3 unitary blocks here:
            (1) the empty block preceding the reset,
            (2) the block inbetween the reset and measurement,
            (3) the block succeeding the measurement
        In this case, unitary_blocks' data is
            {0: [], 1: [H(0)], 2: []}
            with indices corresponding to above labelling.
        Then, the reset dict would be:
            {(0, Qubit(0), RZ): {"preceding": 0, "succeeding": 1}}
        and the measurement dict:
            {(0, Qubit(0), MZ): {"preceding": 1, "succeeding": 2}}
        Where you can see that the reset and measurement dicts simply point to the unitary blocks,
        such that if unitary block 1 is compiled at some point, any changes will be propagated to
        later compilations automatically. If it is updated after compiling resets, the changes
        will be present when measurements are considered.

    The algorithm to collect such data structures from the circuit is as follows:
        For each gate layer:
            For each qubit in layer:
                if it is a special gate:
                    (0) all gates encountered on this qubit so far are the unitary block preceding this gate
                    (1) update the relevant dict: {(layer index, qubit, gate): {"preceding": unitary_block_index}}
                    (2) reset (clear) the gates_encountered dict for this qubit
                    (3) mark current special gate as most recent special gate on this qubit
                    current unitary block may also be a "succeeding" for the most recent special gate on this qubit
                    (4) check most recent special gate on this qubit
                    (5) if exists, update as above but with "succeeding" key
                else:
                    add it to gates encountered on this qubit so far
        If not gate layer, append to none-GateLayer struct.

        do a final update using the remaining encountered gates per qubit, as these
        correspond to gates at the end of the circuit and must be stored as "succeeding"
        of all the last encountered special gates on each qubit.

        if a qubit had no resets, measurements or two-qubit gates on it, then we drop it from the circuit.


    Parameters
    ----------
    circuit : Circuit
        Circuit to parse into a data structure such that
        compilation is possible.

    Returns
    -------
    CompilationData
        A dataclass of the data required for compilation, extracted
        from `circuit`.
    """
    # iterate over the layers
    # {(gate_layer_index, qubit_index, gate_stim_string): {preceding: [], succeeding: []}}
    unitary_blocks: dict[int, list[OneQubitCliffordGate]] = {}
    reset_dict: SpecialGateDict = {}
    meas_dict: SpecialGateDict = {}
    two_q_dict: SpecialGateDict = {}
    non_gate_layers: dict[int, Layer] = {}

    # for each special gate, keep track of any unitaries that have been encountered
    # e.g, for a circuit: q0 = --X--Z--S--RZ--, this would be {q0: [X, Z, S]} up until
    # the reset, at which point those unitaries are set as a unitary block preceding
    # the RZ, then cleared.
    current_gates_per_qubit: dict[Qubit, list[OneQubitCliffordGate]] = {
        q: [] for q in circuit.qubits
    }
    # keep track of the most recent special gate encountered on a qubit. initially set
    # to the identity on qubit -1 at layer -1 as a default value.
    # e.g, after encountering an RZ at layer 0 on Qubit(0), this will be {Qubit(0): (RZ(0), 0)}.
    # if we then later encounter an MZ at layer 5 on Qubit(0), this will be {Qubit(0): (MZ(0), 5)}.
    most_recent_special_gate_per_qubit: dict[Qubit, tuple[Gate, int]] = {
        q: (I(-1), -1) for q in circuit.qubits
    }
    unitary_block_index = 0
    for layer_i, layer in enumerate(circuit.layers):
        # if layer is Circuit, add it to non_gate_layers. this allows us to
        # preserve repeat blocks when unpacking this data structure later
        if isinstance(layer, Circuit):
            # since we've encountered a repeat block, add any trailing unitaries
            # as succeeding blocks to the most recent special gates, and clear
            # the current_gates_per_qubit dict so gates don't carry over beyond
            # the repeat block
            for qubit in circuit.qubits:
                unitary_blocks[unitary_block_index] = current_gates_per_qubit[qubit]
                if not isinstance(most_recent_special_gate_per_qubit[qubit][0], I):
                    dict_to_update = _get_relevant_dict_to_update(
                        most_recent_special_gate_per_qubit[qubit][0],
                        reset_dict,
                        meas_dict,
                        two_q_dict,
                    )
                    dict_to_update[
                        (
                            most_recent_special_gate_per_qubit[qubit][1],
                            qubit,
                            most_recent_special_gate_per_qubit[qubit][0].stim_string,
                        )
                    ]["succeeding"] = unitary_block_index
                else:
                    # this qubit had no meas, reset or 2q gates applied to it
                    continue
                current_gates_per_qubit[qubit] = []
                unitary_block_index += 1
            # need to clear most recent gate per qubit too
            most_recent_special_gate_per_qubit = {
                q: (I(-1), -1) for q in circuit.qubits
            }
            non_gate_layers[layer_i] = _extract_structure_from_circuit(layer)
        elif isinstance(layer, GateLayer):
            # traverse GateLayer
            for gate in layer.gates:
                # if gate is just a unitary, add it and move on
                if isinstance(gate, MPP):
                    msg = "MPP gates not yet supported"
                    raise NotImplementedError(msg)
                if not isinstance(
                    gate, (RX, RY, RZ, MRX, MRZ, MRY, MX, MY, MZ, TwoOperandGate)
                ):
                    current_gates_per_qubit[gate.qubit].append(gate)
                    continue
                # from here we know it is a special gate.
                # load current unitary block based on gates encountered on this qubit so far
                for qubit in gate.qubits:
                    unitary_blocks[unitary_block_index] = current_gates_per_qubit[qubit]

                    # check which of the special gates `gate` is, and update
                    dict_to_update = _get_relevant_dict_to_update(
                        gate, reset_dict, meas_dict, two_q_dict
                    )
                    dict_to_update[(layer_i, qubit, gate.stim_string)] = {
                        "preceding": unitary_block_index
                    }

                    # for this qubit, also add this unitary block as succeeding, if there is
                    # a special gate previously encountered on this qubit
                    if not isinstance(most_recent_special_gate_per_qubit[qubit][0], I):
                        dict_to_update = _get_relevant_dict_to_update(
                            most_recent_special_gate_per_qubit[qubit][0],
                            reset_dict,
                            meas_dict,
                            two_q_dict,
                        )
                        dict_to_update[
                            (
                                most_recent_special_gate_per_qubit[qubit][1],
                                qubit,
                                most_recent_special_gate_per_qubit[qubit][
                                    0
                                ].stim_string,
                            )
                        ]["succeeding"] = unitary_block_index
                    most_recent_special_gate_per_qubit[qubit] = (gate, layer_i)

                    # reset current gates per qubit, update index
                    current_gates_per_qubit[qubit] = []
                    unitary_block_index += 1
        else:
            non_gate_layers[layer_i] = layer

    # after reaching end of circuit, there may be some trailing unitaries that
    # need to be added as `succeeding` blocks.
    # iterate over all qubits to ensure none are missed
    for qubit in circuit.qubits:
        unitary_blocks[unitary_block_index] = current_gates_per_qubit[qubit]
        if not isinstance(most_recent_special_gate_per_qubit[qubit][0], I):
            dict_to_update = _get_relevant_dict_to_update(
                most_recent_special_gate_per_qubit[qubit][0],
                reset_dict,
                meas_dict,
                two_q_dict,
            )
            dict_to_update[
                (
                    most_recent_special_gate_per_qubit[qubit][1],
                    qubit,
                    most_recent_special_gate_per_qubit[qubit][0].stim_string,
                )
            ]["succeeding"] = unitary_block_index
        else:
            # this qubit had no meas, reset or 2q gates applied to it
            continue
        unitary_block_index += 1

    return CompilationData(
        unitary_blocks,
        reset_dict,
        meas_dict,
        two_q_dict,
        non_gate_layers,
        circuit.iterations,
        len(circuit.layers),
    )


def _add_gates_from_unitary_blocks(
    unitary_blocks: dict[int, list[Gate]],
    gate_unitary_block_info: dict[str, int],
    circuit_layers: defaultdict[int, GateLayer],
    layer_index: int,
    unitary_blocks_already_added: set[int],
):
    r"""
    Edit the given `circuit_layers` by adding gates from the unitary blocks to the relevant
    `GateLayer`\ s.

    Parameters
    ----------
    unitary_blocks : Dict[int, List[Gate]]
        Dictionary mapping unitary block indices to lists of single-qubit gates.
    gate_unitary_block_info : Dict[str, int]
        For the given special gate, a dictionary describing which unitary blocks
        precede and succeed it.
    circuit_layers : DefaultDict[int, GateLayer]
        DefaultDictionary storing the GateLayers of the circuit to be constructed.
        Will be edited by adding gates to GateLayers as per the other parameters.
    layer_index : int
        Current layer index, to know which layer indices to add gates to preceding
        and succeeding the special gate.
    unitary_blocks_already_added : Set[int]
        Set to look up which unitary blocks have already been added, so we do not
        add gates more than once.
    """
    # iterate backwards through the preceding block and put in preceding gate layers
    if gate_unitary_block_info["preceding"] not in unitary_blocks_already_added:
        preceding_unitary_block = unitary_blocks[gate_unitary_block_info["preceding"]]
        for i in range(len(preceding_unitary_block)):
            circuit_layers[layer_index - (i + 1)].add_gates(
                preceding_unitary_block[::-1][i]
            )

    # iterate forwards through the succeeding block and put in succeeding gate layers
    if gate_unitary_block_info["succeeding"] not in unitary_blocks_already_added:
        succeeding_unitary_block = unitary_blocks[gate_unitary_block_info["succeeding"]]
        for i, gate in enumerate(succeeding_unitary_block):
            circuit_layers[layer_index + (i + 1)].add_gates(gate)


def _create_circuit_from_compilation_data(
    comp_data: CompilationData, layer_index_lookup: dict[int, int], iterations: int = 1
) -> Circuit:
    """
    The inverse of `_extract_structure_from_circuit` - from a CompilationData object,
    create a `deltakit.circuit.Circuit`.

    Parameters
    ----------
    comp_data : CompilationData
        CompilationData from which to construct a circuit.
    layer_index_lookup : Dict[int, int]
        Dictionary with keys of the original layer indices, and values being their adjusted
        indices accounting for the addition of unitary gates during the compilation process.
    iterations : int
        The number of iterations of the created circuit.

    Returns
    -------
    Circuit
        `deltakit.circuit.Circuit` representation of the circuit described by CompilationData.
    """
    circuit_layers: defaultdict[int, GateLayer] = defaultdict(GateLayer)
    unitary_blocks_already_added: set[int] = set()

    # insert one-qubit special gates
    for special_gate_dict, gate_string_lookup in (
        (comp_data.reset_gates, RESET_GATE_MAPPING),
        (comp_data.measurement_gates, MEASUREMENT_GATE_MAPPING),
    ):
        for gate_info, unitary_block_info in special_gate_dict.items():
            # add special gate to relevant layer
            layer_index, qubit, gate = (
                layer_index_lookup[gate_info[0]],
                gate_info[1],
                gate_info[2],
            )
            circuit_layers[layer_index].add_gates(gate_string_lookup[gate](qubit))

            _add_gates_from_unitary_blocks(
                comp_data.unitary_blocks,
                unitary_block_info,
                circuit_layers,
                layer_index,
                unitary_blocks_already_added,
            )
            unitary_blocks_already_added.add(unitary_block_info["preceding"])
            unitary_blocks_already_added.add(unitary_block_info["succeeding"])

    # insert two-qubit gates
    two_q_gate_dict_as_list = list(comp_data.two_qubit_gates.items())
    for i in range(0, len(two_q_gate_dict_as_list), 2):
        # iterate in pairs
        qubit1_gate_info, qubit1_unitary_block_info = two_q_gate_dict_as_list[i]
        qubit2_gate_info, qubit2_unitary_block_info = two_q_gate_dict_as_list[i + 1]
        layer_index, two_q_gate = (
            layer_index_lookup[qubit1_gate_info[0]],
            TWO_QUBIT_GATE_MAPPING[qubit1_gate_info[2]],
        )
        circuit_layers[layer_index].add_gates(
            two_q_gate(qubit1_gate_info[1], qubit2_gate_info[1])
        )

        # insert unitary blocks
        for unitary_block_info in (
            qubit1_unitary_block_info,
            qubit2_unitary_block_info,
        ):
            _add_gates_from_unitary_blocks(
                comp_data.unitary_blocks,
                unitary_block_info,
                circuit_layers,
                layer_index,
                unitary_blocks_already_added,
            )
            unitary_blocks_already_added.add(unitary_block_info["preceding"])
            unitary_blocks_already_added.add(unitary_block_info["succeeding"])

    # add non-GateLayers
    for layer_index, layer in comp_data.non_gatelayer_layers.items():
        circuit_layers[layer_index_lookup[layer_index]] = layer

    # need to sort circuit_layers, or we may get layers out of order
    return Circuit(dict(sorted(circuit_layers.items())).values(), iterations=iterations)
