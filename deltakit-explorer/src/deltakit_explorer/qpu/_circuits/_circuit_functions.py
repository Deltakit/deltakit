# (c) Copyright Riverlane 2020-2025.
"""
This module consists of individual circuit optimisation functions.
"""


import itertools

from deltakit_circuit import (
    Circuit,
    Detector,
    GateLayer,
    MeasurementRecord,
    NoiseLayer,
    Observable,
    ShiftCoordinates,
)
from deltakit_circuit.gates import I

# According to the purpose and business logic, content of this file
# may violate pylint rules on cyclomatic complexity. Still, code
# complexity requirement should be enforced for the new code.
# pylint: disable=too-many-branches,too-many-nested-blocks,too-many-statements
# pylint: disable=too-many-return-statements,too-many-boolean-expressions


def merge_layers(circuit: Circuit, break_repeat_blocks: bool = False) -> Circuit:
    """
    Merge adjacent gate layers if they act on distinct qubits. Appropriate reindexing
    of measurement records also occurs. Once initial merging has occurred, merging of
    a layer from outside a repeat block with a layer from inside a repeat block will
    be considered. A maximum of one repeat of the circuit may be removed from
    the end of the repeat block, if the break_repeat_blocks option is specified and
    if the next layer after the repeat block is a gate layer. This function requires
    the circuit to have no noise layers.

    Parameters
    ----------
    circuit : Circuit
        Input circuit with layers to be merged.
    break_repeat_blocks : bool, optional
        If True, a maximum of one repeat from the end of a repeat block will be
        "broken out" of the block in order to facilitate merging with a layer
        outside the repeat block. By default, False.

    Returns
    -------
    Circuit
        Circuit with merged layers.
    """

    def _calculate_offset_annotation_layers(
        annotation_layers_with_offsets: list[
            tuple[Detector | Observable | ShiftCoordinates, int]
        ],
    ) -> list[Detector | Observable | ShiftCoordinates]:
        """
        Construct a list of annotations with shifted measurement indices.

        Parameters
        ----------
        annotation_layers_with_offsets : List[
            Union[Detector, Observable, ShiftCoordinates], int]
        ]
            Input annotation layers with their respective offsets to be applied to
            the measurement record measurement indices.

        Returns
        -------
        List[Union[Detector, Observable, ShiftCoordinates]]
            Annotation layers with measurement record indices shifted.
        """

        layers = []
        for annotation, offset in annotation_layers_with_offsets:
            if isinstance(annotation, ShiftCoordinates):
                layers.append(annotation)
            else:
                measurement_records = {
                    MeasurementRecord(measurement_record.lookback_index - offset)
                    for measurement_record in annotation.measurements
                }
                if isinstance(annotation, Detector):
                    new_annotation = Detector(
                        measurements=measurement_records,
                        coordinate=annotation.coordinate,
                    )
                else:
                    new_annotation = Observable(
                        measurements=measurement_records,
                        observable_index=annotation.observable_index,
                    )
                layers.append(new_annotation)

        return layers

    iterations = circuit.iterations

    merged_layers = []
    current_gate_layer = GateLayer()
    current_annotations_and_offsets: list[
        tuple[Detector | Observable | ShiftCoordinates, int]
    ] = []
    for layer in circuit.layers:
        if isinstance(layer, GateLayer):
            # try to merge layer with current_gate_layer
            if current_gate_layer.qubits.isdisjoint(layer.qubits):
                current_gate_layer.add_gates(layer.gates)
                current_annotations_and_offsets = [
                    (annotation, measurement_offset + len(layer.measurement_gates))
                    for (
                        annotation,
                        measurement_offset,
                    ) in current_annotations_and_offsets
                ]
            else:
                merged_layers.append(current_gate_layer)
                merged_layers += _calculate_offset_annotation_layers(
                    current_annotations_and_offsets
                )
                current_gate_layer = GateLayer(layer.gates)
                current_annotations_and_offsets = []
        elif isinstance(layer, Circuit):
            # apply current_gate_layer and annotations before considering repeat block
            if len(current_gate_layer.gates) > 0:
                merged_layers.append(current_gate_layer)
                current_gate_layer = GateLayer()
            merged_layers += _calculate_offset_annotation_layers(
                current_annotations_and_offsets
            )
            current_annotations_and_offsets = []
            merged_layers.append(merge_layers(layer, break_repeat_blocks))
        elif isinstance(layer, NoiseLayer):
            msg = "Layer merge cannot be carried out on a circuit with noise layers."
            raise ValueError(msg)
        else:
            # add annotation to current annotation layers
            current_annotations_and_offsets.append((layer, 0))

    # add remaining layers to end of circuit
    if len(current_gate_layer.gates) != 0:
        merged_layers.append(current_gate_layer)
    merged_layers += _calculate_offset_annotation_layers(
        current_annotations_and_offsets
    )

    if not break_repeat_blocks:
        return Circuit(merged_layers, iterations=iterations)

    # now remove one repeat from the repeat block if it is beneficial to do so
    new_merged_layers = []
    skip_layer = False
    for layer, next_layer in itertools.pairwise(merged_layers):
        if skip_layer:
            skip_layer = False
            continue

        if isinstance(layer, Circuit) and isinstance(next_layer, GateLayer):
            nested_circuit_layers = layer.layers
            gate_layer = GateLayer(next_layer.gates)
            # try to merge nested circuit and following gate layer
            merged_nested_circuit_and_layer = merge_layers(
                Circuit([*nested_circuit_layers, gate_layer]), break_repeat_blocks
            )
            if len(merged_nested_circuit_and_layer.layers) <= len(
                nested_circuit_layers
            ):
                # if merge is successful, remove one repeat from the nested circuit
                new_merged_layers.append(
                    Circuit(nested_circuit_layers, iterations=layer.iterations - 1)
                )
                new_merged_layers += merged_nested_circuit_and_layer.layers
                skip_layer = True
            else:
                new_merged_layers.append(layer)
        else:
            new_merged_layers.append(layer)

    if not skip_layer:
        new_merged_layers.append(merged_layers[-1])

    return Circuit(new_merged_layers, iterations=iterations)


def remove_identities(circuit: Circuit) -> Circuit:
    """
    Remove identity gates from a circuit.

    Parameters
    ----------
    circuit : Circuit
        Circuit from which to remove identity gates.

    Returns
    -------
    Circuit
        Circuit with identity gates removed.
    """
    new_layers = []
    for layer in circuit.layers:
        if isinstance(layer, Circuit):
            new_layer = remove_identities(layer)
            if len(new_layer.layers) > 0:
                new_layers.append(new_layer)
        elif isinstance(layer, GateLayer):
            new_layer = GateLayer(
                [gate for gate in layer.gates if not isinstance(gate, I)]
            )
            if len(new_layer.gates) > 0:
                new_layers.append(new_layer)
        else:
            new_layers.append(layer)

    return Circuit(new_layers, iterations=circuit.iterations)
