# (c) Copyright Riverlane 2020-2025.
"""
This module contains standalone functions which help extract information from
stabilisers.
"""
from collections.abc import Iterable, Sequence

from deltakit_circuit import GateLayer, PauliX, PauliY, PauliZ, Qubit
from deltakit_circuit._basic_maps import PAULI_TO_CP
from deltakit_circuit._qubit_identifiers import PauliGate
from stim import PauliString, Tableau

from deltakit_explorer.codes._stabiliser import Stabiliser


def get_entangling_layer(
    stabilisers: Sequence[Stabiliser],
    layer_index: int,
) -> GateLayer:
    """
    Constructs one layer of entangling gates in the syndrome extraction
    circuit of the stabilisers provided.

    Parameters
    ----------
    stabilisers : Sequence[Stabiliser]
        Stabilisers taking place simultaneously in the syndrome extraction circuit.
    layer_index : int
        The entangling gate layer in the syndrome extraction circuit.

    Returns
    -------
    GateLayer
    """
    ctrl_gate_dict = PAULI_TO_CP

    gates = []
    for stab in stabilisers:
        if (pauli := stab.paulis[layer_index]) is not None:
            two_qubit_gate = ctrl_gate_dict[type(pauli)]
            gates.append(two_qubit_gate(stab.ancilla_qubit, pauli.qubit))

    return GateLayer(gates)


def _transform_stabiliser(
    stabiliser: Stabiliser,
    tableau_qubits: tuple[Qubit],
    transform_tableau: Tableau,
) -> Stabiliser:
    """
    Transforms the stabiliser provided with a unitary Clifford specified by a Stim
    tableau.

    Parameters
    ----------
    stabiliser : Stabiliser
        The stabiliser to transform.
    tableau_qubits : Tuple[Qubit]
        The qubits present in the tableau, indexed in the same way as they appear in
        it.
    transform_tableau : stim.Tableau
        The tableau representing the unitary Clifford to apply to the stabiliser.

    Returns
    -------
    Stabiliser
        The transformed stabiliser.
    """
    output_stabiliser = PauliString(len(tableau_qubits))
    for pauli in stabiliser.paulis:
        if pauli is None:
            continue
        qubit_ind = tableau_qubits.index(pauli.qubit)
        if isinstance(pauli, PauliX):
            output_pauli = transform_tableau.x_output(qubit_ind)
        elif isinstance(pauli, PauliY):
            output_pauli = transform_tableau.y_output(qubit_ind)
        else:
            output_pauli = transform_tableau.z_output(qubit_ind)
        output_stabiliser *= output_pauli
    return Stabiliser(
        paulis=[
            [
                None,
                PauliX(tableau_qubits[ind]),
                PauliY(tableau_qubits[ind]),
                PauliZ(tableau_qubits[ind]),
            ][pauli_num]
            for ind, pauli_num in enumerate(output_stabiliser)
            if pauli_num > 0
        ]
    )


def _get_data_qubits_from_stabilisers(
    stabilisers: Sequence[Sequence[Stabiliser]],
) -> tuple[Qubit, ...]:
    """
    Extracts and returns data qubits from the sequence of stabiliser
    sequences provided.
    """
    return tuple(
        sorted(
            {
                qubit
                for simultaneous_stabilisers in stabilisers
                for stabiliser in simultaneous_stabilisers
                for qubit in stabiliser.data_qubits
            },
            key=lambda qubit: qubit.unique_identifier,
        )
    )


def pauli_gates_to_stim_pauli_string(
    pauli_gates: Iterable[PauliGate | None],
    data_qubit_to_index_lookup: dict[Qubit, int],
) -> PauliString:
    r"""
    Given an iterable of `PauliGate`\ s, express it as a stim.PauliString. To do
    this, a `data_qubit_to_index_lookup` dictionary must also be provided, where
    each data qubit is mapped uniquely to an integer, for the purpose of turning
    `pauli_gates` into a `stim.PauliString`.

    Parameters
    ----------
    pauli_gates : Iterable[Optional[PauliGate]]
        Iterable of `PauliGate`, to be turned into a `stim.PauliString`. Allows
        some of these terms to be None (in which case they are just ignored) such
        that e.g. Stabiliser.paulis may be passed directly into this function.
    data_qubit_to_index_lookup : dict[Qubit, int]
        Dictionary containing a lookup of each data qubit to an integer. This is
        needed since qubit coordinates are usually `Coord2D`, so we must have an
        integer representation for converting to a `stim.PauliString`.

    Returns
    -------
    stim.PauliString
        `stim.PauliString` expression of `pauli_gates`.
    """

    if not all(
        pauli.qubit in data_qubit_to_index_lookup
        for pauli in pauli_gates
        if pauli is not None
    ):
        missing_qubits = {
            pauli.qubit for pauli in pauli_gates if pauli is not None
        }.difference(set(data_qubit_to_index_lookup.keys()))
        msg = (
            "data_qubit_to_index_lookup does not contain entries for"
            f" {missing_qubits} in pauli_gates"
        )
        raise ValueError(msg)
    paulistr = "*".join(
        pauli.stim_identifier + str(data_qubit_to_index_lookup[pauli.qubit])
        for pauli in pauli_gates
        if pauli is not None
    )

    # Adds trailing identity so that the total number of elements in the pauli strings
    # always equals to the number of qubits. This is needed because this trail is present
    # in a Tableau's pauli strings, and two pauli strings which have different trails
    # are not deemed equal by stim. For instance, PauliString("X") != PauliString("X_")
    # Note: The '+ 1' accounts for the +/- sign that gets added in front of the Pauli String
    # when it is initialised.
    return PauliString(
        str(PauliString(paulistr)).ljust(len(data_qubit_to_index_lookup) + 1, "_")
    )
