# (c) Copyright Riverlane 2020-2025.
"""Module containing useful functions to aid in the interaction of decoding graphs and Stim circuits.
"""

import stim
from deltakit_circuit import Circuit, trim_detectors
from deltakit_core.decoding_graphs import (
    DecodingEdge,
    DemParser,
    DetectorCounter,
    FixedWidthBitstring,
    NXDecodingGraph,
    dem_to_decoding_graph_and_logicals,
)


def stim_circuit_to_graph_dem(stim_circuit: stim.Circuit,
                              approximate_disjoint_errors: bool = True
                              ) -> stim.DetectorErrorModel:
    """For a given stim circuit, return the graph-like detector error model.
    If the non-decomposed DEM is graph-like, that will be returned. Otherwise,
    the decomposed DEM will be returned.

    Parameters
    ----------
    stim_circuit : stim.Circuit
        Stim circuit to get the DEM for
    approximate_disjoint_errors : bool, optional
        Iff True, disjoint error approximations will be allowed.
    """
    dem = stim_circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=approximate_disjoint_errors)

    detector_counter = DetectorCounter()
    DemParser(detector_counter, lambda *_: None).parse(dem)

    if detector_counter.max_num_detectors() > 2:
        dem = stim_circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=approximate_disjoint_errors)

    return dem


def parse_stim_circuit(stim_circuit: stim.Circuit,
                       trim_circuit: bool = True,
                       lexical_detectors: bool = True,
                       ) -> tuple[NXDecodingGraph,
                                  list[set[DecodingEdge]],
                                  stim.Circuit]:
    """Parse a Stim file into a decoding graph and the relevant logicals.

    Parameters
    ----------
    stim_circuit : stim.Circuit
        Input Stim circuit to parse.
    trim_circuit : bool, optional
        If True, the output will be trimmed to only include the parts relevant to the
        logicals specified, by default True.
    lexical_detectors : bool, optional
        If True, the detectors in the stim file will be re-ordered to coordinate
        lexical order. No detector will move in a way that changes the meaning
        of the detector. By default True.

    Returns
    -------
    Tuple[NXDecodingGraph, List[Set[DecodingEdge]], stim.Circuit]
        The decoding graph, the logicals, and the Stim circuit. The Stim
        circuit will be unchanged unless `trim_circuit` is set, in which case
        a copy of the Stim circuit with only the trimmed detectors is returned.
    """
    if lexical_detectors:
        circuit = Circuit.from_stim_circuit(stim_circuit)
        circuit.reorder_detectors()
        stim_circuit = circuit.as_stim_circuit()

    dem = stim_circuit_to_graph_dem(stim_circuit)
    graph, logicals = dem_to_decoding_graph_and_logicals(dem)

    if trim_circuit:
        relevant_nodes = graph.get_relevant_nodes(logicals)
        irrelevant_nodes = set(graph.nodes) - relevant_nodes
        stim_circuit = trim_detectors(stim_circuit, irrelevant_nodes)
        dem = stim_circuit_to_graph_dem(stim_circuit)
        graph, logicals = dem_to_decoding_graph_and_logicals(dem)

    return graph, logicals, stim_circuit


def split_measurement_bitstring(
    bitstring: FixedWidthBitstring,
    circuit: stim.Circuit
) -> list[FixedWidthBitstring]:
    """Split a measurement bitstring according to the number of measurements
    in each layer of the circuit.

    Parameters
    ----------
    bitstring : FixedWidthBitstring
        The measurement bitstring.
    circuit : stim.Circuit
        The stim circuit that generated the bitstring.

    Returns
    -------
    List[FixedWidthBitstring]
        The measurement bitstring split so each item in the list
        corresponds to measurements taken in the same layer.
    """
    # flatten circuit to account for any REPEATs in the circuit
    circuit = Circuit.from_stim_circuit(circuit.flattened())

    # starting index of split bitstring
    start_idx = 0
    split_bitstrings: list[FixedWidthBitstring] = []

    for layer in circuit.gate_layers():
        number_of_measurements_in_layer = len(layer.measurement_gates)
        if number_of_measurements_in_layer != 0:
            end_idx = start_idx + number_of_measurements_in_layer
            split_bitstrings.append(bitstring[start_idx:end_idx])
            start_idx = end_idx
    return split_bitstrings
