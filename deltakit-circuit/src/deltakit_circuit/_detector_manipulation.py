# (c) Copyright Riverlane 2020-2025.
"""Functionality for removing a set of detectors from a stim file
using a Detector Error Model's detector indices as input"""

from collections import defaultdict
from collections.abc import Iterable
from warnings import warn

import stim

from deltakit_circuit._annotations._detector import Detector
from deltakit_circuit._circuit import Circuit, Layer


def trim_detectors(
    stim_circuit: stim.Circuit, dem_detectors_to_eliminate: Iterable[int]
) -> stim.Circuit:
    """Uses deltakit_circuit to remove a given set of detectors from a
    stim circuit and returns the new stim circuit

    Parameters
    ----------
    stim_circuit : stim.Circuit
        A stim circuit from which detectors are to be removed

    detectors_to_eliminate : Iterable[int]
        A set of detectors to be removed. A detector to be removed
        should be specified via an integers in a list. The integers
        associated with a detector must be determined in the same way as
        integers are assigned to detectors via
        stim.Circuit.detector_error_model.

        Importantly, if a stim file specifies a detector within a repeat
        block then to remove this detector you have to list all indices
        that that detector would be associated with in its corresponding
        detector error model, else the detector will not be eliminated
        and a warning will be raised. For example, given

        ..
            repeat 3 {
                DETECTOR(5, 0) rec[-6] rec[-2]
                DETECTOR(7, 0) rec[-5] rec[-1]
            }

        each detector will occur three times in a DEM and have the
        indices [0, 2, 4] and [1, 3, 5] respectively. So, if removal
        of the first detector is done by specifying only index [0] for
        elimination a warning will be raised and the detector will
        not be removed as this is ambiguous.

        Any number of nested repeat blocks is supported and this
        explains some of the code's complexity.

    Returns
    -------
    stim.Circuit
        A copy of the original stim circuit without the detectors
        specified via detectors_to_eliminate
    """
    deltakit_circuit_circuit = Circuit.from_stim_circuit(stim_circuit)
    ordered_detector_calls = _get_ordered_detector_calls(deltakit_circuit_circuit)
    condensed_detector_calls = _condense_detector_calls(ordered_detector_calls)
    logical_detectors_to_eliminate = _get_detectors_to_remove(
        dem_detectors_to_eliminate, condensed_detector_calls
    )
    deltakit_circuit_circuit, _ = _trim_detectors(
        deltakit_circuit_circuit, logical_detectors_to_eliminate
    )
    return deltakit_circuit_circuit.as_stim_circuit()


def _get_ordered_detector_calls(deltakit_circuit_circuit: Circuit) -> list[Detector]:
    """Unravels a deltakit_circuit circuit to put each Detector in a
    list each time it would be called at execution time
    """
    ordered_detector_calls: list[Detector] = []
    for _ in range(deltakit_circuit_circuit.iterations):
        for layer in deltakit_circuit_circuit.layers:
            if isinstance(layer, Circuit):
                ordered_detector_calls.extend(_get_ordered_detector_calls(layer))
            if isinstance(layer, Detector):
                ordered_detector_calls.append(layer)
    return ordered_detector_calls


def _condense_detector_calls(
    ordered_detector_calls: Iterable[Detector],
) -> list[list[int]]:
    """Takes a list of deltakit_circuit detector objects and associates
    each detector call with the complete set of corresponding DEM indices
    """
    detector_to_dem_indices = defaultdict(list)
    for index, detector in enumerate(ordered_detector_calls):
        detector_to_dem_indices[id(detector)].append(index)
    return list(detector_to_dem_indices.values())


def _get_detectors_to_remove(
    detectors_to_eliminate: Iterable[int],
    condensed_detector_calls: Iterable[Iterable[int]],
) -> list[int]:
    indices = []
    for stim_index, call_indices in enumerate(condensed_detector_calls):
        if all(x in detectors_to_eliminate for x in call_indices):
            indices.append(stim_index)
        elif any(x in detectors_to_eliminate for x in call_indices):
            warn(
                "Detector is being specified for removal "
                "via an incomplete list of DEM detector IDs. "
                f"To remove this detector specify indices {call_indices}.",
                RuntimeWarning,
                stacklevel=3,
            )
    return indices


def _trim_detectors(
    deltakit_circuit_circuit: Circuit,
    detectors_to_eliminate: Iterable[int],
    detector_count: int = 0,
) -> tuple[Circuit, int]:
    layers: list[Layer] = []

    for layer in deltakit_circuit_circuit.layers:
        if isinstance(layer, Circuit):
            inner_layer, detector_count = _trim_detectors(
                layer, detectors_to_eliminate, detector_count
            )
            layers.append(Circuit(inner_layer, layer.iterations))
        elif isinstance(layer, Detector):
            if detector_count not in detectors_to_eliminate:
                layers.append(layer)
            detector_count += 1
        else:
            layers.append(layer)

    return Circuit(layers), detector_count
