# (c) Copyright Riverlane 2020-2025.
"""
This module aggregates functions to manipulate `deltakit_circuit.Circuit` objects.
These include native gate compilation, noise introduction,
and various optimisations.
This module is not currently public; this __init__.py file is a historical artefact
and can be removed, adjusting imports within other `deltakit_explorer` modules
accordingly.
"""


from deltakit_explorer.qpu._circuits._circuit_functions import (
    merge_layers,
    remove_identities,
)
from deltakit_explorer.qpu._circuits._parallelise_circuits import (
    parallelise_disjoint_circuits,
    parallelise_same_length_circuits,
)
from deltakit_explorer.qpu._circuits._tableau_compile_functions import (
    _compile_measurement_to_native_gates_plus_unitaries,
    _compile_or_exchange_unitary_block,
    _compile_reset_and_meas_to_native_gates,
    _compile_reset_to_native_gates_plus_unitaries,
    _compile_two_qubit_gate_to_target,
    _compile_two_qubit_gates_to_native_gates,
    compile_circuit_to_native_gates,
)
from deltakit_explorer.qpu._circuits._tableau_functions import (
    CZ_TO_GATE_DICT,
    CZSWAP_TO_GATE_DICT,
    GATE_TO_CZ_DICT,
    GATE_TO_CZSWAP_DICT,
    MEAS_COMPILATION_LOOKUP_DICT,
    RESET_COMPILATION_LOOKUP_DICT,
    CompilationData,
    _create_circuit_from_compilation_data,
    _extract_structure_from_circuit,
    _get_compilation_dict,
    _get_compilation_with_measurement_after_unitaries,
    _get_compilation_with_projectors_before_unitaries,
    _get_compilation_with_two_qubit_gates,
    _is_identity_like,
)

__all__ = [s for s in dir() if not s.startswith("_")]
