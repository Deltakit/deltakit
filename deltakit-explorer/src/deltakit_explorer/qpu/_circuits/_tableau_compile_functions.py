from collections.abc import Callable, Sequence

from deltakit_circuit import Circuit, Qubit
from deltakit_circuit.gates import (
    CXSWAP,
    CZSWAP,
    ISWAP,
    ISWAP_DAG,
    MEASUREMENT_GATE_MAPPING,
    ONE_QUBIT_GATE_MAPPING,
    RESET_GATE_MAPPING,
    TWO_QUBIT_GATE_MAPPING,
    Gate,
    SymmetricTwoQubitGate,
)

from deltakit_explorer.qpu._circuits._circuit_functions import merge_layers
from deltakit_explorer.qpu._native_gate_set import (
    NativeGateSetAndTimes,
    OneQubitCliffordGate,
    TwoOperandGate,
)

from ._tableau_functions import (
    CZ_TO_GATE_DICT,
    CZSWAP_TO_GATE_DICT,
    GATE_TO_CZ_DICT,
    GATE_TO_CZSWAP_DICT,
    MEAS_COMPILATION_LOOKUP_DICT,
    RESET_COMPILATION_LOOKUP_DICT,
    CompilationData,
    EquivalentTableauDict,
    SpecialGateDict,
    TableauDict,
    TwoQubitGateCompilationLookupDict,
    TwoQubitGateDictEntry,
    _create_circuit_from_compilation_data,
    _extract_structure_from_circuit,
    _get_compilation_dict,
    _get_compilation_with_measurement_after_unitaries,
    _get_compilation_with_projectors_before_unitaries,
    _get_compilation_with_two_qubit_gates,
    _get_tableau_as_key,
    _get_tableau_from_sequence_of_1q_gates,
    _get_tableau_key_from_sequence_of_gates,
)


def _compile_to_native_gates_plus_unitaries(
    gate_info: tuple[int, Qubit, str],
    preceding_ub: list[OneQubitCliffordGate],
    succeeding_ub: list[OneQubitCliffordGate],
    available_native_gates: set[Gate],
    available_single_qubit_gates: set[OneQubitCliffordGate],
    compilation_lookup_dict: dict[
        Gate, dict[Gate, tuple[list[OneQubitCliffordGate], list[OneQubitCliffordGate]]]
    ],
    special_gate_mapping: dict[str, Gate],
    comp_dict: TableauDict,
    up_to_paulis: bool,
) -> tuple[list[OneQubitCliffordGate], list[OneQubitCliffordGate], str]:
    """
    Given a reset or measurement gate and its surrounding unitaries, attempt to compile it
    to the native gate set, by trying each available reset/measurement gate and trying to
    compile the resulting unitaries into native gates. If this fails, a ValueError is raised.

    Parameters
    ----------
    gate_info : Tuple[int, Qubit, str]
        Current gate's info, from the special gate dictionary, detailing the layer index,
        qubit and stim_string of the gate.
    preceding_ub : List[OneQubitCliffordGate]
        Unitary block preceding the gate.
    succeeding_ub : List[OneQubitCliffordGate]
        Unitary block succeeding the gate.
    available_native_gates : Set[Gate]
        Available native reset/measurement gates to try to compile to.
    available_single_qubit_gates : Set[OneQubitCliffordGate]
        Available single qubit gates to compile the unitary blocks to.
    compilation_lookup_dict : Dict[
        Gate, Dict[Gate, Tuple[List[OneQubitCliffordGate], List[OneQubitCliffordGate]]]
    ]
        Dictionary detailing how to compile from one reset/measurement gate to another.
    special_gate_mapping : Dict[str, Gate]
        Dictionary to lookup the stim_string to gate type.
    comp_dict : TableauDict
        Compilation dictionary, with tableaus as keys and lists of unitary
        gates as values.
    up_to_paulis : bool
        Whether to compute compilation up to pauli terms or not.

    Returns
    -------
    Tuple[List[OneQubitCliffordGate], List[OneQubitCliffordGate], str]
        A tuple containing the preceding and succeeding unitary blocks after
        compilation to native gates, and the stim_string of the gate compiled to.
    """
    # try compiling to a native gate. If we run out, the compilation is not
    # possible, so throw an error.
    while len(available_native_gates) >= 0:
        try:
            target_gate = available_native_gates.pop()
        except KeyError as ke:
            msg = (
                "Unable to compile to provided native reset and measurement gates,"
                " please try changing the native gate set."
            )
            raise ValueError(msg) from ke

        # look up the compilation from current gate to target native gate.
        # if the entry is not in the dictionary, the compilation is not
        # supported, so try the next native gate.
        try:
            unitaries_to_add = compilation_lookup_dict[
                special_gate_mapping[gate_info[2]]
            ][target_gate]
        except KeyError:
            continue

        updated_preceding_ub, updated_succeeding_ub = [], []
        for unitary_block, ub_is_preceding in (
            (preceding_ub, True),
            (succeeding_ub, False),
        ):
            # try compiling the updated unitary_block to native gates. if any contain
            # a non-native gate after compilation, stop and try the next target gate.
            # otherwise, stop, as compilation succeeded
            if ub_is_preceding:
                updated_preceding_ub = _compile_or_exchange_unitary_block(
                    unitary_block + [g(gate_info[1]) for g in unitaries_to_add[0]],
                    comp_dict,
                    up_to_paulis,
                )
            else:
                updated_succeeding_ub = _compile_or_exchange_unitary_block(
                    [g(gate_info[1]) for g in unitaries_to_add[1]] + unitary_block,
                    comp_dict,
                    up_to_paulis,
                )

        # if the gates in the unitary blocks are a subset of the native gates, we're done
        if set({type(g) for g in updated_preceding_ub + updated_succeeding_ub}) <= set(
            available_single_qubit_gates
        ):
            break
    return updated_preceding_ub, updated_succeeding_ub, target_gate.stim_string


def _compile_reset_to_native_gates_plus_unitaries(
    gate_info: tuple[int, Qubit, str],
    preceding_ub: list[OneQubitCliffordGate],
    succeeding_ub: list[OneQubitCliffordGate],
    native_gate_set: NativeGateSetAndTimes,
    comp_dict: TableauDict,
    up_to_paulis: bool,
) -> tuple[list[OneQubitCliffordGate], list[OneQubitCliffordGate], str]:
    return _compile_to_native_gates_plus_unitaries(
        gate_info,
        preceding_ub,
        succeeding_ub,
        set(native_gate_set.reset_gates.keys()),
        set(native_gate_set.one_qubit_gates.keys()),
        RESET_COMPILATION_LOOKUP_DICT,
        RESET_GATE_MAPPING,
        comp_dict,
        up_to_paulis,
    )


def _compile_measurement_to_native_gates_plus_unitaries(
    gate_info: tuple[int, Qubit, str],
    preceding_ub: list[OneQubitCliffordGate],
    succeeding_ub: list[OneQubitCliffordGate],
    native_gate_set: NativeGateSetAndTimes,
    comp_dict: TableauDict,
    up_to_paulis: bool,
) -> tuple[list[OneQubitCliffordGate], list[OneQubitCliffordGate], str]:
    return _compile_to_native_gates_plus_unitaries(
        gate_info,
        preceding_ub,
        succeeding_ub,
        set(native_gate_set.measurement_gates.keys()),
        set(native_gate_set.one_qubit_gates.keys()),
        MEAS_COMPILATION_LOOKUP_DICT,
        MEASUREMENT_GATE_MAPPING,
        comp_dict,
        up_to_paulis,
    )


def _compile_or_exchange_unitary_block(
    unitary_block: list[OneQubitCliffordGate],
    comp_dict: TableauDict,
    up_to_paulis: bool,
) -> list[OneQubitCliffordGate]:
    """
    Given a unitary block in the form of a list of one qubit gates that are possibly not
    native gates, attempt to compile the unitary block to a shorter set of gates, so
    that the unitary block is expressed in terms of only native gates.

    Parameters
    ----------
    unitary_block : List[OneQubitCliffordGate]
        Unitary block to attempt to shorten or exchange for an equivalent expression in terms
        of just native gates.
    comp_dict : TableauDict
        Dictionary containing the shortest discovered expression for a given tableau.
    up_to_paulis : bool
        Boolean specifying whether to perform the compilation up to pauli terms or not.

    Returns
    -------
    List[OneQubitCliffordGate]
        Updated unitary block.
    """
    # get tableau of unitary block
    ub_tab = _get_tableau_as_key(
        _get_tableau_from_sequence_of_1q_gates([g.stim_string for g in unitary_block]),
        up_to_paulis=up_to_paulis,
    )

    # compile or exchange
    if ub_tab in comp_dict:
        # if we can compile to something shorter, do that
        unitary_block = [
            ONE_QUBIT_GATE_MAPPING[g](unitary_block[0].qubit) for g in comp_dict[ub_tab]
        ]
    return unitary_block


def _compile_reset_and_meas_to_native_gates(
    comp_data: CompilationData,
    native_gate_set: NativeGateSetAndTimes,
    comp_dict: TableauDict,
    up_to_paulis: bool,
    layer_index_lookup: dict[int, int],
) -> tuple[CompilationData, dict[int, int]]:
    """
    Given a circuit in the form of CompilationData, attempt to compile the
    reset and measurement gates into native gates, possibly with extra unitary
    gates inserted into the surrounding unitary blocks. We then attempt to compile
    or exchange the unitary blocks for something in terms of native gates. If this
    is not possible, the compilation has failed, so we throw a ValueError.
    After the unitary blocks have been updated, layer_index_lookup is modified
    to allow room for any extra unitaries that have been added. For instance,
    when creating CompilationData, say a reset and two qubit gate are separated
    by one gate layer in between.

    If we then compiled the reset and consequently require two unitaries following
    the reset, we need to add another layer in between the reset and two qubit gate.
    This information is tracked in layer_index_lookup, where the positive delta of
    unitary blocks before and after compilation is added to the succeeding layer
    indices to make room for the new unitaries. In the example mentioned above,
    if the reset was at layer 10, and the two qubit gate at layer 12, after
    compilation the two qubit gate would be at layer 13, and the dictionary would
    have entry {12: 13} to reflect this. Then when the circuit is reconstructed from
    CompilationData, the layer indices recorded before compilation are looked up in
    the layer_index_lookup dictionary to add the gates at the right layer indices
    such that there is room for the unitaries.

    Parameters
    ----------
    comp_data : CompilationData
        Circuit to compile in the form of CompilationData.
    native_gate_set : NativeGateSetAndTimes
        Native gate set to compile to.
    comp_dict : TableauDict
        Dictionary containing the shortest discovered expression for a given tableau.
    up_to_paulis : bool
        Boolean specifying whether to perform the compilation up to pauli terms or not.
    layer_index_lookup : Dict[int, int]
        Dictionary with keys of the original layer indices, and values being their adjusted
        indices accounting for the addition of unitary gates during the compilation process.

    Returns
    -------
    Tuple[CompilationData, Dict[int, int]]
        A tuple containing the compiled circuit in the form of CompilationData,
        and the index lookup dictionary that has as keys the original layer index
        and as values the adjusted index of that layer after taking into account
        the addition of unitary gates in between layers due to compilation.
    """
    # Takes the same form of SpecialGateDict, but will be updated
    # to reflect the compilation. E.g, originally we may have
    # (0, Qubit(0), "RX") for an RX on qubit 0 in the first gate layer,
    # but after compiling to RZ, this will change to (0, Qubit(0), "RZ").
    # We make new dictionaries because dictionary keys are immutable.
    compiled_reset_gates: SpecialGateDict = {}
    compiled_measurement_gates: SpecialGateDict = {}
    for special_dict, dict_to_update, compile_function, gate_lookup, native_gates in (
        (
            comp_data.reset_gates,
            compiled_reset_gates,
            _compile_reset_to_native_gates_plus_unitaries,
            RESET_GATE_MAPPING,
            set(native_gate_set.reset_gates.keys()),
        ),
        (
            comp_data.measurement_gates,
            compiled_measurement_gates,
            _compile_measurement_to_native_gates_plus_unitaries,
            MEASUREMENT_GATE_MAPPING,
            set(native_gate_set.measurement_gates.keys()),
        ),
    ):
        for gate_info, gate_ubs in special_dict.items():
            if gate_lookup[gate_info[2]] in native_gates:
                dict_to_update[gate_info] = gate_ubs
                continue

            # Try compiling to the current target native gate by seeing how the
            # surrounding unitary blocks change
            ub1_index, ub2_index = gate_ubs["preceding"], gate_ubs["succeeding"]
            ub1, ub2, compiled_gate = compile_function(
                gate_info,
                comp_data.unitary_blocks[ub1_index],
                comp_data.unitary_blocks[ub2_index],
                native_gate_set,
                comp_dict,
                up_to_paulis,
            )
            # check whether the preceding or succeeding unitaries have got longer
            # than in the original circuit, in which case we must update the layer indices
            # of special gates following the current special gate
            for ub_index, new_ub, start_delta in (
                (ub1_index, ub1, 0),
                (ub2_index, ub2, 1),
            ):
                # start_delta is 0 when we consider the preceding unitary block, so
                # that we adjust the layer index of the current gate. It is then 1
                # for the succeeding, since we do not need to adjust the layer index
                # of the current gate if only the succeeding unitaries have gotten longer
                delta = len(new_ub) - len(comp_data.unitary_blocks[ub_index])
                if delta > 0:
                    for i in range(
                        gate_info[0] + start_delta, max(layer_index_lookup.keys()) + 1
                    ):
                        layer_index_lookup[i] += delta

            # update relevant parts of comp_data
            comp_data.unitary_blocks[ub1_index] = ub1
            comp_data.unitary_blocks[ub2_index] = ub2
            dict_to_update[(gate_info[0], gate_info[1], compiled_gate)] = gate_ubs

    comp_data.reset_gates = compiled_reset_gates
    comp_data.measurement_gates = compiled_measurement_gates
    return comp_data, layer_index_lookup


def _compile_two_qubit_gate_to_target(
    current_gate: TwoOperandGate,
    target_gate: TwoOperandGate,
    qubit1_entry: TwoQubitGateDictEntry,
    qubit2_entry: TwoQubitGateDictEntry,
    gate_to_intermediate_rep_dict: TwoQubitGateCompilationLookupDict,
    int_rep_to_target_dict: TwoQubitGateCompilationLookupDict,
    unitary_blocks: dict[int, list],
) -> tuple[
    list[OneQubitCliffordGate],
    list[OneQubitCliffordGate],
    list[OneQubitCliffordGate],
    list[OneQubitCliffordGate],
]:
    r"""
    Given a two-qubit gate, and a target two-qubit gate, compute the updated
    unitary blocks required to compile the current gate to the target gate.
    This is done by first working out the expression of `current_gate` in
    terms of the intermediate representation (CZ or CZSWAP) + unitaries,
    then going from the intermediate representation to `target_gate` with
    further unitaries.

    Parameters
    ----------
    current_gate : TwoOperandGate
        The current gate, to be compiled from.
    target_gate : TwoOperandGate
        The target gate, to be compiled to. I.e, `current_gate` will be
        expressed in terms of `target_gate`.
    qubit1_entry : TwoQubitGateDictEntry
        The entry of CompilationData.two_qubit_gates corresponding to
        qubit 1 of the two qubit gate.
    qubit2_entry : TwoQubitGateDictEntry
        The entry of CompilationData.two_qubit_gates corresponding to
        qubit 2 of the two qubit gate.
    gate_to_int_rep_dict : TwoQubitGateCompilationLookupDict
        Dictionary providing lookup from `current_gate` to the intermediate
        representation (CZ or CZSWAP) for compiling.
        Should be either GATE_TO_CZ_DICT or GATE_TO_CZSWAP_DICT.
    int_rep_to_target_dict : TwoQubitGateCompilationLookupDict
        Dictionary providing lookup from the intermediate
        representation (CZ or CZSWAP) to `target_gate`.
        Should be either CZ_TO_GATE_DICT or CZSWAP_TO_GATE_DICT.
    unitary_blocks : Dict[int, List]
        The CompilationData.unitary_blocks dictionary, providing information
        on the indices of unitary blocks and their contents.

    Returns
    -------
    Tuple[
        List[OneQubitCliffordGate],
        List[OneQubitCliffordGate],
        List[OneQubitCliffordGate],
        List[OneQubitCliffordGate],
    ]
        A tuple containing, in order,
        the unitary blocks corresponding to:
                [0]: Before qubit 1
                [1]: After qubit 1
                [2]: Before qubit 2
                [3]: After qubit 2
        Each in the form of a list of `OneQubitCliffordGate`\ s.
    """
    qubit1_info, qubit1_ubs = qubit1_entry
    qubit2_info, qubit2_ubs = qubit2_entry
    ub0, ub1, ub2, ub3 = (
        unitary_blocks[qubit1_ubs["preceding"]],
        unitary_blocks[qubit1_ubs["succeeding"]],
        unitary_blocks[qubit2_ubs["preceding"]],
        unitary_blocks[qubit2_ubs["succeeding"]],
    )

    # First, replace current gate with intermediate representation by
    # adding appropriate unitaries
    try:
        ubs = gate_to_intermediate_rep_dict[current_gate]
    except KeyError as ke:
        msg = "Current gate not present in gate_to_intermediate_rep_dict dictionary"
        raise ValueError(msg) from ke
    ub0 = ub0 + [g(qubit1_info[1]) for g in ubs[0]]
    ub1 = [g(qubit1_info[1]) for g in ubs[1]] + ub1
    ub2 = ub2 + [g(qubit2_info[1]) for g in ubs[2]]
    ub3 = [g(qubit2_info[1]) for g in ubs[3]] + ub3

    # Then, replace the intermediate representation with the target gate by
    # adding appropriate unitaries, taking into account that the intermediate
    # representation gate or this gate may need to be reversed
    try:
        ubs_target = int_rep_to_target_dict[target_gate]
    except KeyError as ke:
        msg = (
            "Cannot compile between groups -"
            f" {current_gate.stim_string} to {target_gate.stim_string} not supported"
        )
        raise ValueError(msg) from ke

    ub0 = ub0 + [g(qubit1_info[1]) for g in ubs_target[0]]
    ub1 = [g(qubit1_info[1]) for g in ubs_target[1]] + ub1
    ub2 = ub2 + [g(qubit2_info[1]) for g in ubs_target[2]]
    ub3 = [g(qubit2_info[1]) for g in ubs_target[3]] + ub3
    return ub0, ub1, ub2, ub3


def _compile_two_qubit_gates_to_native_gates(
    comp_data: CompilationData,
    native_gate_set: NativeGateSetAndTimes,
    comp_dict: TableauDict,
    up_to_paulis: bool,
    layer_index_lookup: dict[int, int],
) -> tuple[CompilationData, dict[int, int]]:
    """
    Given a circuit in the form of CompilationData, attempt to compile the
    two-qubit gates into native gates, possibly with extra unitary
    gates inserted into the surrounding unitary blocks.
    Along the way, `layer_index_lookup` is updated to account for
    extra gate layers needed to fit in the unitaries required for
    computation.
    If compilation to native gates not possible, throws a ValueError.

    Parameters
    ----------
    comp_data : CompilationData
        Circuit to compile in the form of CompilationData.
    native_gate_set : NativeGateSetAndTimes
        Native gate set to compile to.
    comp_dict : TableauDict
        Dictionary containing the shortest discovered expression for a given tableau.
    up_to_paulis : bool
        Boolean specifying whether to perform the compilation up to pauli terms or not.
    layer_index_lookup : Dict[int, int]
        Dictionary with keys of the original layer indices, and values being their adjusted
        indices accounting for the addition of unitary gates during the compilation process.

    Returns
    -------
    Tuple[CompilationData, Dict[int, int]]
        Tuple containing the compiled circuit in the form of CompilationData,
        and the updated `layer_index_lookup`.
    """
    if len(comp_data.two_qubit_gates) == 0:
        return comp_data, layer_index_lookup
    # Updated version of the CompilationData.two_qubit_gates dictionary that will contain
    # the newly compiled versions of each two qubit gate. To be filled in as the compilation
    # proceeds.
    new_two_qubit_dict: SpecialGateDict = {}
    dict_as_list = list(comp_data.two_qubit_gates.items())
    # for each two-qubit gate, attempt to compile it to any of the available native gates
    for i in range(0, len(dict_as_list), 2):
        qubit1_entry: TwoQubitGateDictEntry = dict_as_list[i]
        qubit2_entry: TwoQubitGateDictEntry = dict_as_list[i + 1]
        current_gate = TWO_QUBIT_GATE_MAPPING[qubit1_entry[0][2]]
        if current_gate in native_gate_set.two_qubit_gates:
            # if current gate in native gates, add it as-is to new dict and move on
            new_two_qubit_dict[qubit1_entry[0]] = qubit1_entry[1]
            new_two_qubit_dict[qubit2_entry[0]] = qubit2_entry[1]
            continue

        # set relevant lookup dicts
        if current_gate in (ISWAP, ISWAP_DAG, CXSWAP, CZSWAP):
            gate_to_intm_dict = GATE_TO_CZSWAP_DICT
            intm_to_target_dict = CZSWAP_TO_GATE_DICT
        else:
            gate_to_intm_dict = GATE_TO_CZ_DICT
            intm_to_target_dict = CZ_TO_GATE_DICT

        set_of_possible_gates = set(native_gate_set.two_qubit_gates.keys())
        while len(set_of_possible_gates) >= 0:
            # for each possible native two-qubit gate
            try:
                target_gate = set_of_possible_gates.pop()
            except KeyError as ke:
                msg = (
                    "Unable to compile to provided native two-qubit gates,"
                    " please try changing the native gate set."
                )
                raise ValueError(msg) from ke

            # compile the two qubit gate, by getting the new unitary blocks
            updated_unitaries = _compile_two_qubit_gate_to_target(
                current_gate,
                target_gate,
                qubit1_entry,
                qubit2_entry,
                gate_to_intm_dict,
                intm_to_target_dict,
                comp_data.unitary_blocks,
            )

            # test if the new unitary blocks can be compiled to something in terms of native gates.
            # if not, the compilation failed, so try the next native gate
            compilation_successful = True
            compiled_updated_unitaries = []
            for ub in updated_unitaries:
                compiled_unitary_block = _compile_or_exchange_unitary_block(
                    ub,
                    comp_dict,
                    up_to_paulis,
                )

                # if a non-native gate remains after compilation, cannot compiled to this 2q gate
                if not set({type(g) for g in compiled_unitary_block}) <= set(
                    native_gate_set.one_qubit_gates.keys()
                ):
                    compilation_successful = False
                    break
                compiled_updated_unitaries.append(compiled_unitary_block)

            if compilation_successful:
                # record change in layer indices, based on the unitary block with the
                # highest delta
                ub1_delta = len(compiled_updated_unitaries[0]) - len(
                    comp_data.unitary_blocks[qubit1_entry[1]["preceding"]]
                )
                ub3_delta = len(compiled_updated_unitaries[2]) - len(
                    comp_data.unitary_blocks[qubit2_entry[1]["preceding"]]
                )
                max_delta_before_gate = max(ub1_delta, ub3_delta)

                ub2_delta = len(compiled_updated_unitaries[1]) - len(
                    comp_data.unitary_blocks[qubit1_entry[1]["succeeding"]]
                )
                ub4_delta = len(compiled_updated_unitaries[3]) - len(
                    comp_data.unitary_blocks[qubit2_entry[1]["succeeding"]]
                )
                max_delta_after_gate = max(ub2_delta, ub4_delta)

                # for unitary blocks before, adjust the layer of the gate itself.
                # for unitary blocks after, adjust the layers following the gate, but not the gate itself.
                for start_delta, delta in (
                    (0, max_delta_before_gate),
                    (1, max_delta_after_gate),
                ):
                    if delta > 0:
                        for i in range(
                            qubit1_entry[0][0] + start_delta,
                            max(layer_index_lookup.keys()) + 1,
                        ):
                            layer_index_lookup[i] += delta

                # update unitary blocks
                comp_data.unitary_blocks[
                    qubit1_entry[1]["preceding"]
                ] = compiled_updated_unitaries[0]
                comp_data.unitary_blocks[
                    qubit1_entry[1]["succeeding"]
                ] = compiled_updated_unitaries[1]
                comp_data.unitary_blocks[
                    qubit2_entry[1]["preceding"]
                ] = compiled_updated_unitaries[2]
                comp_data.unitary_blocks[
                    qubit2_entry[1]["succeeding"]
                ] = compiled_updated_unitaries[3]

                new_two_qubit_dict[
                    (
                        qubit1_entry[0][0],
                        qubit1_entry[0][1],
                        target_gate.stim_string,
                    )
                ] = qubit1_entry[1]
                new_two_qubit_dict[
                    (
                        qubit2_entry[0][0],
                        qubit2_entry[0][1],
                        target_gate.stim_string,
                    )
                ] = qubit2_entry[1]
                break
    comp_data.two_qubit_gates = new_two_qubit_dict
    return comp_data, layer_index_lookup


def _compile_comp_data(
    comp_data: CompilationData,
    layer_index_lookup: dict[int, int],
    native_gate_set: NativeGateSetAndTimes,
    comp_dict: TableauDict,
    gate_exchange_dict: EquivalentTableauDict,
    up_to_paulis: bool,
    allow_terms_to_mutate: bool,
    allow_terms_to_multiply: bool,
) -> Circuit:
    """
    Given a CompilationData object, and the corresponding relevant data, compile
    this CompilationData to native gates and return a deltakit_circuit.Circuit.

    Parameters
    ----------
    comp_data : CompilationData
        Circuit to compile, in CompilationData form.
    layer_index_lookup : Dict[int, int]
        Accompanying layer index lookup dictionary, used to ensure
        gates added for compilation do not clash in layers.
    native_gate_set : NativeGateSetAndTimes
        Native gate set to try compiling to.
    comp_dict : TableauDict
        Compilation dictionary containing the shortest discovered gate
        expression for a given tableau.
    gate_exchange_dict : EquivalentTableauDict
        Dictionary containing equivalent expressions in terms of different
        (but still native) gates of each tableau in comp_dict. Used for
        two-qubit gate compilation where equivalent gate expressions
        for a given tableau may give better compilations.
    up_to_paulis: bool
        Specify whether compilation should happen only up to Pauli terms.
        By default, False.
    allow_terms_to_mutate : bool
        For two qubit gate compilation, will allow the case of pulling a single-qubit Pauli
        term through a two-qubit gate such that the resulting (possibly two-qubit) Pauli has
        a different term on either qubit.
        E.g., pulling an X through a CZ from the first qubit mutates into a Z term
        on the second qubit.
        By default, True.
    allow_terms_to_multiply : bool
        For two qubit gate compilation, will allow the case of pulling a term through a
        two-qubit gate such that a second term on the other qubit is created. E.g., pulling
        an X through a CX from the first qubit mutates into an X on the first qubit and an
        X on the second qubit.
        By default, True.

    Returns
    -------
    deltakit_circuit.Circuit
        Circuit compiled from the given CompilationData.
    """
    # compile repeat blocks first - call this function for each CompilationData
    # in the non_gatelayer_layers dict.
    for layer_index, layer in comp_data.non_gatelayer_layers.items():
        if isinstance(layer, CompilationData):
            layer_index_lookup = {
                i: i for i in range(max(comp_data.num_layers, layer.num_layers))
            }
            comp_data.non_gatelayer_layers[layer_index] = _compile_comp_data(
                layer,
                layer_index_lookup,
                native_gate_set,
                comp_dict,
                gate_exchange_dict,
                up_to_paulis,
                allow_terms_to_mutate,
                allow_terms_to_multiply,
            )

    # step 1: compile reset and measurement gates to native reset and
    # measurement gates + some unitaries.
    comp_data, layer_index_lookup = _compile_reset_and_meas_to_native_gates(
        comp_data,
        native_gate_set,
        comp_dict,
        up_to_paulis,
        layer_index_lookup,
    )

    # step 2: compile two-qubit gates to native
    comp_data, layer_index_lookup = _compile_two_qubit_gates_to_native_gates(
        comp_data,
        native_gate_set,
        comp_dict,
        up_to_paulis,
        layer_index_lookup,
    )

    # step 2a: go through all unitary blocks and compile to native, reducing
    # where possible. This is because, in the previous steps,
    # if the unitary blocks preceding a two qubit gate are empty,
    # the following blocks will not be looked at. Similarly, if a reset or measurement
    # is part of the native gates, its unitaries are also ignored.
    for unitary_block_index, unitary_block in comp_data.unitary_blocks.items():
        ub_tab = _get_tableau_as_key(
            _get_tableau_from_sequence_of_1q_gates(
                [g.stim_string for g in unitary_block]
            ),
            up_to_paulis=up_to_paulis,
        )
        if ub_tab in comp_dict and len(comp_dict[ub_tab]) < len(unitary_block):
            comp_data.unitary_blocks[unitary_block_index] = [
                ONE_QUBIT_GATE_MAPPING[g](unitary_block[0].qubit)
                for g in comp_dict[ub_tab]
            ]

    # step 3: compile unitary blocks succeeding measurements
    for meas_gate_info, ub_info in comp_data.measurement_gates.items():
        unitaries_after = comp_data.unitary_blocks[ub_info["succeeding"]]
        updated_unitaries_after = _get_compilation_with_projectors_before_unitaries(
            comp_dict,
            unitaries_after,
            MEASUREMENT_GATE_MAPPING[meas_gate_info[2]],
            up_to_paulis=up_to_paulis,
        )
        comp_data.unitary_blocks[ub_info["succeeding"]] = [
            ONE_QUBIT_GATE_MAPPING[g](meas_gate_info[1].unique_identifier)
            for g in updated_unitaries_after
        ]

    # step 4: compile two qubit gates and surrounding unitaries
    two_qubit_dict_as_list = list(comp_data.two_qubit_gates.items())
    # define a convenience lambda for compiling the unitaries to natives,
    # for use in flipping in the two-qubit gates
    unitary_blocks_to_native: Callable[  # noqa: E731
        [Sequence[OneQubitCliffordGate]], list[OneQubitCliffordGate]
    ] = lambda ub: [
        ONE_QUBIT_GATE_MAPPING[g]
        for g in comp_dict[
            _get_tableau_key_from_sequence_of_gates(ub, up_to_paulis=up_to_paulis)
        ]
    ]
    for i in range(0, len(two_qubit_dict_as_list), 2):
        qubit1_gate_info, qubit1_unitary_info = two_qubit_dict_as_list[i]
        qubit2_gate_info, qubit2_unitary_info = two_qubit_dict_as_list[i + 1]

        ub1_before_reducing = [
            type(g) for g in comp_data.unitary_blocks[qubit1_unitary_info["preceding"]]
        ]
        ub2_before_reducing = [
            type(g) for g in comp_data.unitary_blocks[qubit1_unitary_info["succeeding"]]
        ]
        ub3_before_reducing = [
            type(g) for g in comp_data.unitary_blocks[qubit2_unitary_info["preceding"]]
        ]
        ub4_before_reducing = [
            type(g) for g in comp_data.unitary_blocks[qubit2_unitary_info["succeeding"]]
        ]

        two_qubit_gate_to_compile = TWO_QUBIT_GATE_MAPPING[qubit1_gate_info[2]]
        (
            ub1_after_reducing,
            ub2_after_reducing,
            ub3_after_reducing,
            ub4_after_reducing,
        ) = _get_compilation_with_two_qubit_gates(
            two_qubit_gate_to_compile,
            comp_dict,
            ub1_before_reducing,
            ub2_before_reducing,
            ub3_before_reducing,
            ub4_before_reducing,
            gate_exchange_dict,
            up_to_paulis=up_to_paulis,
            allow_terms_to_mutate=allow_terms_to_mutate,
            allow_terms_to_multiply=allow_terms_to_multiply,
        )
        if not issubclass(two_qubit_gate_to_compile, SymmetricTwoQubitGate):
            # If non-symmetric two-qubit gate, try flipping the orientation
            # of the gate to get a better compilation. This is achieved by
            # swapping the positions of the unitary blocks. We can do this
            # with one-qubit gates, using the CZ compilation of the 2q gate.

            # Note that CXSWAP is the only non-symmetric gate from the iSWAP
            # class, at least in stim currently.
            if two_qubit_gate_to_compile in (CXSWAP,):
                gate_to_intm_dict = GATE_TO_CZSWAP_DICT
                intm_to_target_dict = CZSWAP_TO_GATE_DICT
            else:
                gate_to_intm_dict = GATE_TO_CZ_DICT
                intm_to_target_dict = CZ_TO_GATE_DICT

            # get the unitaries used for compiling to and from CZ, but we will
            # swap the orientation of the adj. gates to swap the orientation
            # of the two-qubit gate
            # the unitaries in `intm_to_target_dict`` are guaranteed to be the adjoint
            # of those in `gate_to_intm_dict`` by construction
            ub1, ub2, ub3, ub4 = gate_to_intm_dict[two_qubit_gate_to_compile]
            ub1_adj, ub2_adj, ub3_adj, ub4_adj = intm_to_target_dict[
                two_qubit_gate_to_compile
            ]
            try:
                # If G_12 is the native (non-symmetric) 2-qubit gate, and G_21 is its
                # flipped version, then since G = u_1 \otimes u_3 CZ_12 u_2 \otimes u_4,
                # we can calculate that G = (u_1 \otimes u_3) (u_3adj \otimes u_1adj) CZ_12
                # (u_4adj \otimes u_2adj) (u_2 \otimes u_4).
                # this might produce a KeyError if the gates for swapping orientation
                # do not have a native gate expression
                ub3_with_flip_unitaries = unitary_blocks_to_native(
                    ub3_before_reducing + ub3 + ub1_adj
                )
                ub4_with_flip_unitaries = unitary_blocks_to_native(
                    ub2_adj + ub4 + ub4_before_reducing
                )
                ub1_with_flip_unitaries = unitary_blocks_to_native(
                    ub1_before_reducing + ub1 + ub3_adj
                )
                ub2_with_flip_unitaries = unitary_blocks_to_native(
                    ub4_adj + ub2 + ub2_before_reducing
                )
                flipping_unitaries_available = True
            except KeyError:
                # in this case, we cannot change the orientation of the gate,
                # since the unitaries to do so do not have an equivalent
                # expression in the provided native gates, so skip that part.
                flipping_unitaries_available = False
            if flipping_unitaries_available:
                (
                    ub3_after_reducing_flipped,
                    ub4_after_reducing_flipped,
                    ub1_after_reducing_flipped,
                    ub2_after_reducing_flipped,
                ) = _get_compilation_with_two_qubit_gates(
                    two_qubit_gate_to_compile,
                    comp_dict,
                    ub3_with_flip_unitaries,
                    ub4_with_flip_unitaries,
                    ub1_with_flip_unitaries,
                    ub2_with_flip_unitaries,
                    gate_exchange_dict,
                    up_to_paulis=up_to_paulis,
                    allow_terms_to_mutate=allow_terms_to_mutate,
                    allow_terms_to_multiply=allow_terms_to_multiply,
                )
                # if flipping gave a better compilation, reverse the direction
                # of the gate
                if len(ub1_after_reducing_flipped) + len(
                    ub3_after_reducing_flipped
                ) < len(ub1_after_reducing) + len(ub3_after_reducing):
                    (
                        ub1_after_reducing,
                        ub2_after_reducing,
                        ub3_after_reducing,
                        ub4_after_reducing,
                    ) = (
                        ub1_after_reducing_flipped,
                        ub2_after_reducing_flipped,
                        ub3_after_reducing_flipped,
                        ub4_after_reducing_flipped,
                    )
                    del comp_data.two_qubit_gates[qubit1_gate_info]
                    del comp_data.two_qubit_gates[qubit2_gate_info]
                    comp_data.two_qubit_gates[qubit2_gate_info] = qubit2_unitary_info
                    comp_data.two_qubit_gates[qubit1_gate_info] = qubit1_unitary_info

        # only ub2 or ub4 can potentially be longer, as we cannot add gates to ub1 or ub3
        delta = max(
            len(ub2_after_reducing) - len(ub2_before_reducing),
            len(ub4_after_reducing) - len(ub4_before_reducing),
        )
        if delta > 0:
            for i in range(qubit1_gate_info[0] + 1, max(layer_index_lookup.keys()) + 1):
                layer_index_lookup[i] += delta

        comp_data.unitary_blocks[qubit1_unitary_info["preceding"]] = [
            g(qubit1_gate_info[1].unique_identifier) for g in ub1_after_reducing
        ]
        comp_data.unitary_blocks[qubit1_unitary_info["succeeding"]] = [
            g(qubit1_gate_info[1].unique_identifier) for g in ub2_after_reducing
        ]
        comp_data.unitary_blocks[qubit2_unitary_info["preceding"]] = [
            g(qubit2_gate_info[1].unique_identifier) for g in ub3_after_reducing
        ]
        comp_data.unitary_blocks[qubit2_unitary_info["succeeding"]] = [
            g(qubit2_gate_info[1].unique_identifier) for g in ub4_after_reducing
        ]

    # step 5: compile unitaries before measurements
    for meas_gate_info, meas_gate_ub_info in comp_data.measurement_gates.items():
        updated_ub = _get_compilation_with_measurement_after_unitaries(
            comp_dict,
            comp_data.unitary_blocks[meas_gate_ub_info["preceding"]],
            MEASUREMENT_GATE_MAPPING[meas_gate_info[2]],
            up_to_paulis=up_to_paulis,
        )
        comp_data.unitary_blocks[meas_gate_ub_info["preceding"]] = [
            ONE_QUBIT_GATE_MAPPING[g](meas_gate_info[1].unique_identifier)
            for g in updated_ub
        ]

    # step 6: compile unitaries before and after resets
    for reset_gate_info, reset_gate_ub_info in comp_data.reset_gates.items():
        comp_data.unitary_blocks[reset_gate_ub_info["preceding"]] = []
        unitaries_after = comp_data.unitary_blocks[reset_gate_ub_info["succeeding"]]
        updated_unitaries_after = _get_compilation_with_projectors_before_unitaries(
            comp_dict,
            unitaries_after,
            RESET_GATE_MAPPING[reset_gate_info[2]],
            up_to_paulis=up_to_paulis,
        )
        comp_data.unitary_blocks[reset_gate_ub_info["succeeding"]] = [
            ONE_QUBIT_GATE_MAPPING[g](reset_gate_info[1].unique_identifier)
            for g in updated_unitaries_after
        ]

    return _create_circuit_from_compilation_data(
        comp_data, layer_index_lookup, iterations=comp_data.iterations
    )


def compile_circuit_to_native_gates(
    circuit: Circuit,
    native_gate_set: NativeGateSetAndTimes,
    up_to_paulis: bool = False,
    allow_terms_to_mutate: bool = True,
    allow_terms_to_multiply: bool = True,
) -> Circuit:
    """
    Compile a deltakit_circuit.Circuit to native gates and reduce the single qubit gate count
    where possible.

    Params
    ------
    circuit : Circuit
        deltakit_circuit.Circuit to compile.
    native_gate_set : NativeGateSetAndTimes
        Native gate set to compile to.
    up_to_paulis: bool, optional
        Specify whether compilation should happen only up to Pauli terms.
        By default, False.
    allow_terms_to_mutate : bool, optional
        For two qubit gate compilation, will allow the case of pulling a single-qubit Pauli
        term through a two-qubit gate such that the resulting (possibly two-qubit) Pauli has
        a different term on either qubit.
        E.g, pulling an X through a CZ from the first qubit mutates into a Z term
        on the second qubit.
        By default, True.
    allow_terms_to_multiply : bool, optional
        For two qubit gate compilation, will allow the case of pulling a term through a
        two-qubit gate such that a second term on the other qubit is created. E.g, pulling
        an X through a CX from the first qubit mutates into an X on the first qubit and an
        X on the second qubit.
        By default, True.

    Returns
    -------
    Circuit
        Compiled circuit.
    """
    # get compilation dictionary and gate exchange dict
    comp_dict, gate_exchange_dict = _get_compilation_dict(
        native_gate_set, up_to_paulis=up_to_paulis
    )

    # get CompilationData form of circuit
    comp_data = _extract_structure_from_circuit(circuit)
    layer_index_lookup = {i: i for i in range(comp_data.num_layers)}

    return merge_layers(
        _compile_comp_data(
            comp_data,
            layer_index_lookup,
            native_gate_set,
            comp_dict,
            gate_exchange_dict,
            up_to_paulis,
            allow_terms_to_mutate,
            allow_terms_to_multiply,
        )
    )
