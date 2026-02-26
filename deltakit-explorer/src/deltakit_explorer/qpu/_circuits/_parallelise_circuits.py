# (c) Copyright Riverlane 2020-2025.
"""
This module includes functions merging multiple circuits into a
single circuit to be executed on a single QPU in parallel.
"""


import numpy as np
from deltakit_circuit import (
    Circuit,
    Detector,
    GateLayer,
    Observable,
    Qubit,
    ShiftCoordinates,
)

# According to the purpose and business logic, content of this file
# may violate pylint rules on cyclomatic complexity. Still, code
# complexity requirement should be enforced for the new code.
# pylint: disable=too-many-branches,too-many-nested-blocks,too-many-statements
# pylint: disable=too-many-return-statements,too-many-boolean-expressions


def parallelise_disjoint_circuits(circuits: list[Circuit]) -> Circuit:
    """
    Parallelise a list of circuits, each of which must act on distinct qubits.
    We assume it is preferable to perform gates of the same type in the same
    layer and thus this is done if possible, but not if it would involve
    making the total number of gate layers in the parallelised circuit larger
    than the maximum number of gate layers in any of the input circuits.
    None of the circuits may contain NoiseLayers. Only one circuit may
    contain annotation layers and only one circuit may contain nested Circuit
    layers - these need not be the same circuit. If one circuit contains
    annotations, the other circuits may not contain measurement gates. The
    circuits must also have the same number of iterations.

    Example
    -------
    Input:
        Circuit 1:
            Layer 1: CNOT(0, 1)
            Layer 2: H(1), H(2)
        Circuit 2:
            Layer 1: H(3)

    Output:
        Circuit:
            Layer 1: CNOT(0, 1)
            Layer 2: H(1), H(2), H(3)


    Parameters
    ----------
    circuits : List[Circuit]
        List of circuits to be parallelised.

    Returns
    -------
    Circuit
        Parallelised circuit.
    """
    if len(circuits) == 0:
        return Circuit()

    # Check for unequal numbers of iterations
    iterations = circuits[0].iterations
    if any(circuit.iterations != iterations for circuit in circuits):
        msg = "Circuits to be parallelised must have the same number of iterations."
        raise ValueError(msg)

    def _any_annotations(circuit: Circuit) -> bool:
        for layer in circuit.layers:
            if isinstance(layer, (Detector, Observable, ShiftCoordinates)):
                return True
            if isinstance(layer, Circuit) and _any_annotations(layer):
                return True
        return False

    def _any_circuit_layers(circuit: Circuit) -> bool:
        return any(isinstance(layer, Circuit) for layer in circuit.layers)

    def _any_measurements(circuit: Circuit) -> bool:
        if len(circuit.measurement_gates) > 0:
            return True
        return any(
            _any_measurements(layer)
            for layer in circuit.layers
            if isinstance(layer, Circuit)
        )

    # Check if any circuit contains NoiseLayers
    if any(circuit.is_noisy for circuit in circuits):
        msg = "Circuits to be parallelised may not contain NoiseLayers."
        raise ValueError(msg)

    # Check which circuits contain annotations
    any_annotations = [_any_annotations(circuit) for circuit in circuits]

    if sum(any_annotations) == 1:
        annotations_circ = any_annotations.index(True)
        if any(
            _any_measurements(circuit)
            for circuit in circuits
            if circuit != circuits[annotations_circ]
        ):
            msg = (
                "If one circuit to be parallelised contains annotations, "
                "no other circuit can contain measurements."
            )
            raise ValueError(msg)
    elif sum(any_annotations) > 1:
        msg = "Only one circuit to be parallelised can contain annotations."
        raise ValueError(msg)

    # Check how many circuits contain a nested circuit
    if sum(_any_circuit_layers(circuit) for circuit in circuits) > 1:
        msg = "Only one circuit to be parallelised can contain a nested Circuit."
        raise ValueError(msg)

    # In future, this can be simplified to the length of non-recursive gate_layers
    circuit_lengths = [
        len([layer for layer in circuit.layers if isinstance(layer, GateLayer)])
        for circuit in circuits
    ]
    total_length = max(circuit_lengths)

    # Get layers for longest circuit
    max_circ_ind = circuit_lengths.index(total_length)
    max_circ = circuits[max_circ_ind]

    # Initialise lists to store the layers and separately the gate types in each layer
    parallelised_layers = []
    parallelised_layers_gate_types = []

    # Add gate types from longest circuit to gate type list and non-GateLayers to
    # layer list
    for layer in max_circ.layers:
        if isinstance(layer, GateLayer):
            parallelised_layers.append(GateLayer())
            parallelised_layers_gate_types.append(
                {type(gate) for gate in layer.gates}
            )
        else:
            parallelised_layers.append(layer)
            parallelised_layers_gate_types.append(set())

    # Initialise set to record qubits
    qubits: set[Qubit] = set()

    # Loop over circuits
    for icirc, (circuit, new_length) in enumerate(zip(circuits, circuit_lengths)):
        # update qubits
        new_qubits = circuit.qubits
        if len(qubits.union(new_qubits)) != len(qubits) + len(new_qubits):
            msg = "Circuits to be parallelised do not act on distinct qubits."
            raise ValueError(msg)
        qubits = qubits.union(new_qubits)

        # set variable to record previous layer in parallelised circuit into which
        # gates were added or where an entire layer was inserted
        par_previous_layer_ind = -1

        # set variable to record the gate layer index of the layer to be inserted
        new_gate_layer_ind = -1

        # set variable to record the gate layer index in the parallelised circuit
        par_gate_layer_ind = -1

        for new_layer in circuit.layers:
            if not isinstance(new_layer, GateLayer):
                par_previous_layer_ind += 1
                if icirc != max_circ_ind:
                    parallelised_layers.insert(par_previous_layer_ind, new_layer)
                    parallelised_layers_gate_types.insert(par_previous_layer_ind, set())
            else:
                new_gate_layer_ind += 1
                for par_layer_ind in range(
                    par_previous_layer_ind + 1, len(parallelised_layers)
                ):
                    par_layer = parallelised_layers[par_layer_ind]
                    par_layer_gate_types = parallelised_layers_gate_types[par_layer_ind]
                    if not isinstance(par_layer, GateLayer):
                        continue

                    # check if number of gate layers left is equal to number available or if
                    # all gates are already in layer
                    par_gate_layer_ind += 1
                    if (
                        new_length - new_gate_layer_ind
                        == total_length - par_gate_layer_ind
                    ) or {
                        type(gate) for gate in new_layer.gates
                    } <= par_layer_gate_types:
                        # add gates to layer
                        par_layer.add_gates(new_layer.gates)

                        # record gate types
                        if icirc != max_circ_ind:
                            par_layer_gate_types.update(
                                {type(gate) for gate in new_layer.gates}
                            )
                        par_previous_layer_ind += par_layer_ind - par_previous_layer_ind
                        break

    return Circuit(parallelised_layers)


def parallelise_same_length_circuits(circuits: list[Circuit]) -> Circuit:
    """
    Parallelise a list of circuits, each of which must be the same length and
    act on distinct qubits in a particular layer. The circuits must also consist
    of GateLayers only.

    Parameters
    ----------
    circuits : List[Circuit]
        List of circuits to be parallelised.

    Returns
    -------
    Circuit
        Parallelised circuit.
    """
    if len(circuits) == 0:
        return Circuit()

    # Check for unequal numbers of iterations
    iterations = circuits[0].iterations
    if any(circuit.iterations != iterations for circuit in circuits):
        msg = "Circuits to be parallelised must have the same number of iterations."
        raise ValueError(msg)

    if any(
        not all(isinstance(layer, GateLayer) for layer in circuit.layers)
        for circuit in circuits
    ):
        msg = "Circuits can only be parallelised if they contain only GateLayers."
        raise ValueError(msg)

    circuit_lengths = [len(circuit.layers) for circuit in circuits]
    num_layers = circuit_lengths[0]
    if not np.all(np.array(circuit_lengths) == num_layers):
        msg = "Circuits must all be the same length."
        raise ValueError(msg)

    # Begin parallelised circuit construction with first circuit
    parallelised_circuit = Circuit(
        [GateLayer(layer.gates) for layer in circuits[0].layers]
    )

    # Loop over remaining circuits
    for circuit in circuits[1:]:
        for ilayer, layer in enumerate(circuit.layers):
            parallelised_circuit.layers[ilayer].add_gates(layer.gates)

    return parallelised_circuit
