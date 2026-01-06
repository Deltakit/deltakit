# (c) Copyright Riverlane 2020-2025.
"""
This module contains standalone functions related to detectors used to generating
stabiliser code circuits.
"""
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable as IterableC
from collections.abc import Sequence
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from deltakit_circuit import (
    Circuit,
    Coordinate,
    Detector,
    MeasurementRecord,
    Qubit,
    ShiftCoordinates,
)

from deltakit_explorer.codes._stabiliser import Stabiliser

if TYPE_CHECKING:
    from deltakit_explorer.codes._css._css_stage import CSSStage


def _get_coordinate_from_ancilla_qubit(
    ancilla_qubit: Qubit | None,
) -> Coordinate | None:
    """
    Extracts and returns the coordinate of the qubit provided. If these coordinates
    are not available (i.e. if the qubit provided is None or if the qubit's unique
    identifier is not a set of coordinates), then returns None instead.

    Before returning the coordinates, initialises an extra coordinate dimension to 0,
    (representing time) such that the coordinate returned readily adopts a detector
    coordinate format.

    Parameters
    ----------
    ancilla_qubit : Qubit, optional
        The qubit of which to extract the coordinate. This function is meant for an
        ancilla qubit from a Stabiliser, which is why it handles the cases where the
        qubit is None or its unique identifier is not a Coordinate.

    Returns
    -------
    Coordinate | None
    """
    if ancilla_qubit is not None:
        unique_id = ancilla_qubit.unique_identifier
        qubit_coord = (
            tuple(unique_id) if isinstance(unique_id, IterableC) else (unique_id,)
        )
        if all(
            isinstance(coord, (float, int, np.floating, np.integer))
            for coord in qubit_coord
        ):
            full_coord = (*qubit_coord, 0)
            return Coordinate(*full_coord)

    return None


def _get_coordinate_from_data_qubits(
    data_qubits: list[Qubit],
) -> Coordinate | None:
    """
    Computes and returns a coordinate which is the average in each coordinate
    dimension of qubits' coordinates provided. If any of these qubits' coordinate is
    not valid (i.e. if any unique identifier is not a set of coordinates), then
    returns None instead.

    Before returning the coordinates, initialises an extra coordinate dimension to 0,
    (representing time) such that the coordinate returned readily adopts a detector
    coordinate format.

    Parameters
    ----------
    data_qubits : List[Qubit], optional
        The qubits from which to compute the coordinate. This function is meant for
        data qubits from a Stabiliser, which is why it handles the cases where the
        qubits' unique identifiers are not Coordinates.

    Returns
    -------
    Coordinate | None
    """
    qubit_coords = [
        (
            tuple(qubit.unique_identifier)
            if isinstance(qubit.unique_identifier, IterableC)
            else (qubit.unique_identifier,)
        )
        for qubit in data_qubits
    ]

    # if any elements are not floats or ints, a valid coordinate cannot be
    # constructed
    if any(
        any(not isinstance(coord, (float, int)) for coord in qubit_coord)
        for qubit_coord in qubit_coords
    ):
        return None

    # if the data qubit coordinates are not all of the same length, a valid
    # detector coordinate cannot be constructed
    if any(
        len(qubit_coord) != len(qubit_coords[0]) for qubit_coord in qubit_coords[1:]
    ):
        return None
    qubit_coord_len = len(qubit_coords[0])

    full_coord = (
        *tuple(
            float(np.mean([qubit_coord[i] for qubit_coord in qubit_coords]))
            for i in range(qubit_coord_len)
        ), 0)
    return Coordinate(*full_coord)


def _calculate_detector_coordinates(
    stabilisers: Sequence[Stabiliser],
) -> tuple[Coordinate, ...]:
    """
    Calculate detector coordinates based on the stabilisers used in the stage.

    Where possible, the coordinate will be taken from the ancilla qubit unique
    identifier. If this is not possible, an attempt will be made to form the
    coordinate from averaging the data qubit unique identifiers.

    If, due to averaging the data qubit unique identifier, multiple
    coordinates are equal, small perturbations are applied.

    However, if it is not possible to form coordinates of the same dimension from
    all stabilisers, or there are further clashes after this process, the coordinates
    will be formed by indexing the stabilisers by their order of input.

    Parameters
    ----------
    stabilisers : Sequence[Stabiliser]
        The stabilisers from which we wish to calculate detector coordinates.

    Returns
    -------
    Tuple[Coordinate, ...]
        Detector coordinates for the input stabilisers, with the ordering the same
        as in the input.
    """

    detector_coords: list[Coordinate] = []
    coord_len: int = -1
    for stabiliser in stabilisers:
        # attempt to get a coordinate from the ancilla qubit
        detector_coord = _get_coordinate_from_ancilla_qubit(stabiliser.ancilla_qubit)
        if (
            detector_coord is not None
            # check if detector coordinate length is consistent with previous lengths
            and (len(detector_coords) == 0 or coord_len == len(detector_coord))
        ):
            detector_coords.append(detector_coord)
            coord_len = len(detector_coord)
            continue

        # attempt to get a coordinate from the data qubits
        detector_coord = _get_coordinate_from_data_qubits(
            [pauli.qubit for pauli in stabiliser.paulis if pauli is not None]
        )
        if detector_coord is None:
            break

        # check if detector coordinate length is consistent with previous lengths
        if len(detector_coords) != 0 and len(detector_coord) != coord_len:
            break

        detector_coords.append(detector_coord)
        coord_len = len(detector_coord)

    else:
        # check for repeats
        for detector_coord in detector_coords:
            if (detector_count := detector_coords.count(detector_coord)) > 1:
                indices = [
                    i
                    for i in range(len(detector_coords))
                    if detector_coords[i] == detector_coord
                ]
                # spread repeated coordinates along their first index
                for i_index, index in enumerate(indices):
                    first_coord = (
                        detector_coord[0] - 0.3 + 0.6 * i_index / (detector_count - 1)
                    )
                    full_coord = (first_coord, *tuple(coord for coord in detector_coord[1:]))
                    detector_coords[index] = Coordinate(*full_coord)

        # check again for repeats as some shifted coordinates with different original
        # coordinates may now overlap
        for detector_coord in detector_coords:
            if detector_coords.count(detector_coord) > 1:
                break
        else:
            return tuple(detector_coords)

    return tuple(Coordinate(i, 0) for i in range(len(stabilisers)))


def _get_coordinate_shifts(detectors: list[Detector]) -> list[ShiftCoordinates]:
    """
    Given a list of detectors, computes coordinate shifts which increase the last
    coordinate dimension by 1. If the detectors provided have coordinates of
    different number of dimensions, then a coordinate shift for each is added.
    A warning is raised in that case.

    Parameters
    ----------
    detectors : List[Detector]
        The list of detectors to use to generate the shift coordinate instructions.

    Returns
    -------
    List[ShiftCoordinates]
        A list of the shift coordinate instructions.
    """

    detector_num_dimensions = [
        len(detector.coordinate)
        for detector in detectors
        if detector.coordinate is not None
    ]

    # obtaining the unique elements in this way to conserve the order
    unique_num_dimensions = list(dict.fromkeys(detector_num_dimensions))

    if len(unique_num_dimensions) > 1:
        warnings.warn(
            "Detector coordinates have different dimensions and thus "
            + "coordinates may shift in multiple directions."
        )

    return [
        ShiftCoordinates((0,) * (num_dimensions - 1) + (1,))
        for num_dimensions in unique_num_dimensions
    ]


def _get_between_round_detectors(
    detector_coordinates: Sequence[Coordinate],
    num_additional_measurements: int = 0,
) -> list[Detector]:
    """
    For a scenario where multiple rounds of syndrome extraction of a same group of
    stabilisers are performed, compute and return the detectors which compare
    measurements lying in different rounds.

    Each detector returned is associated with the measurement (or measurement pair
    across rounds) of the ancilla qubit of one of the stabilisers. We therefore
    obtain one detector per stabiliser. Our detectors consist of measurement pairs
    from two adjacent rounds.
    Parameters
    ----------
    detector_coordinates : Sequence[Coordinate, ...]
        A list of the detector coordinates associated with each of the stabilisers
        measured in each syndrome extraction round, ordered in the same way.
    num_additional_measurements : int, optional
        The number of additional measurements taking place at the end of each round of
        syndrome extraction.

    Returns
    -------
    List[Detector]
        Between-round detectors for the stabilisers.
    """
    if len(detector_coordinates) == 0:
        return []

    num_measurements = len(detector_coordinates) + num_additional_measurements

    detectors: list[Detector | ShiftCoordinates] = []

    for i, detector_coord in enumerate(detector_coordinates):
        lookback_index = i - num_measurements
        detector_mmts = [MeasurementRecord(lookback_index)]

        detector_mmts.append(MeasurementRecord(lookback_index - num_measurements))

        detectors.append(Detector(detector_mmts, detector_coord))

    return detectors


def get_between_round_detectors_and_coordinate_shifts(
    detector_coordinates: Sequence[Coordinate],
    num_additional_measurements: int = 0,
) -> list[Detector | ShiftCoordinates]:
    """
    For a scenario where multiple rounds of syndrome extraction of a same group of
    stabilisers are performed, compute and return the detectors which compare
    measurements lying in different rounds.

    Each detector returned is associated with the measurement (or measurement pair
    across rounds) of the ancilla qubit of one of the stabilisers. We therefore
    obtain one detector per stabiliser. Our detectors consist of measurement pairs
    from two adjacent rounds.

    Note a ShiftCoordinate instruction which increases the last detector coordinate
    dimension (i.e. time) by 1 is appended to the end of the detectors list returned.

    Parameters
    ----------
    detector_coordinates : Sequence[Coordinate]
        A list of the detector coordinates associated with each of the stabilisers
        measured in each syndrome extraction round, ordered in the same way.
    num_additional_measurements : int, optional
        The number of additional measurements taking place at the end of each round of
        syndrome extraction.

    Returns
    -------
    List[Union[Detector, ShiftCoordinates]]
        Between-round detectors for the stabilisers along with a ShiftCoordinate instruction
        that increases the last detector coordinate dimension.
    """
    detectors = _get_between_round_detectors(
        detector_coordinates, num_additional_measurements
    )
    shifts = _get_coordinate_shifts(detectors)

    return detectors + shifts


def _get_joint_sub_super_stabilisers_ind(
    previous_stabilisers: Sequence[Stabiliser],
    current_stabilisers: Sequence[Stabiliser],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Exhaustively search for one element in previous_stabilisers/current_stabilisers and
    one or more elements from current_stabilisers/previous_stabilisers such that the
    product of the latter is the former. The method collects the indices of these
    elements, e.g.
        - ((i_prev,),(i_curr,j_curr,...)) corresponds to one element of
            previous_stabilisers that can be obtained as a product of elements from
            current_stabilisers, or
        - ((i_prev,j_prev,...),(i_curr)) corresponds to one element of
            current_stabilisers that can be obtained as a product of elements from
            previous_stabilisers,

    Parameters
    ----------
    previous_stabilisers : Sequence[Stabiliser]
        Pauli products present in previous CSSStage.
    current_stabilisers : Sequence[Stabiliser]
        Pauli products present in current CSSStage.

    Returns
    -------
    List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
        The list collecting the stabiliser outcome comparisons.
    """
    # First build dictionaries of the comparisons
    joint_stabilisers_ind: dict[int, int] = {}
    sub_stabilisers_ind: dict[int, list[int]] = defaultdict(list)
    super_stabilisers_ind: dict[int, list[int]] = defaultdict(list)

    for (ind_prev, stab_prev), (ind_curr, stab_curr) in itertools.product(
        enumerate(previous_stabilisers), enumerate(current_stabilisers)
    ):
        if stab_prev.operator_repr == stab_curr.operator_repr:
            joint_stabilisers_ind[ind_prev] = ind_curr
        elif stab_prev.operator_repr < stab_curr.operator_repr:
            sub_stabilisers_ind[ind_curr].append(ind_prev)
        elif stab_prev.operator_repr > stab_curr.operator_repr:
            super_stabilisers_ind[ind_prev].append(ind_curr)

    # Check if the small Paulis multiply together to the big Pauli.
    for ind_curr in list(sub_stabilisers_ind.keys()):
        indices_prev = sub_stabilisers_ind[ind_curr]
        previous_stabiliser_product = reduce(
            lambda x, y: x * y,
            [previous_stabilisers[ind_prev] for ind_prev in indices_prev],
        )
        if (
            current_stabilisers[ind_curr].operator_repr != previous_stabiliser_product.operator_repr
        ):
            del sub_stabilisers_ind[ind_curr]

    for ind_prev in list(super_stabilisers_ind.keys()):
        indices_curr = super_stabilisers_ind[ind_prev]
        current_stabiliser_product = reduce(
            lambda x, y: x * y,
            [current_stabilisers[ind_curr] for ind_curr in indices_curr],
        )
        if (
            previous_stabilisers[ind_prev].operator_repr != current_stabiliser_product.operator_repr
        ):
            del super_stabilisers_ind[ind_prev]
    # Convert these dictionaries into the list format
    # fmt: off
    full_index_list: list[tuple[tuple[int, ...], tuple[int, ...]]] = (
        [
            ((ind_prev,), (ind_curr,))
            for ind_prev, ind_curr in sorted(
                joint_stabilisers_ind.items(), key=lambda item: item[0]
            )
        ]
        + [
            (tuple(indices_prev), (ind_curr,))
            for ind_curr, indices_prev in sorted(
                sub_stabilisers_ind.items(), key=lambda item: item[0]
            )
        ]
        + [
            ((ind_prev,), tuple(indices_curr,))
            for ind_prev, indices_curr in sorted(
                super_stabilisers_ind.items(), key=lambda item: item[0]
            )
        ]
    )
    # fmt: on
    return full_index_list


def _get_stage_transition_detectors(
    previous_stage: "CSSStage", current_stage: "CSSStage"
) -> list[Detector]:
    """
    Return a list containing all detectors for transitioning between the
    stages previous_stage and current_stage.

    Parameters
    ----------
    previous_stage : CSSStage
        A CSSStage from which we transition. For instance, for patch growing
        previous_stage would be the stage where we measure a small patch for some
        rounds.
    current_stage : CSSStage
        The CSSStage into which we intend to transition. For instance, for patch
        growing current_stage would be the stage where we measure the bigger patch for
        some rounds.

    Returns
    -------
    Circuit
        The transition detectors.
    """
    detectors = []
    previous_stabilisers = (
        previous_stage.ordered_stabilisers + previous_stage.resets_as_stabilisers
    )
    current_stabilisers = (
        current_stage.measurements_as_stabilisers + current_stage.ordered_stabilisers
    )
    current_stabilisers_before = (
        current_stage.measurements_as_stabilisers + current_stage.stabilisers_before
    )

    # Do comparison with current_stabilisers_before, e.g. if current_stage does a
    # transversal H or SWAP, we need to compare with the stabilisers before the
    # transversal gates.
    joint_sub_super_stabilisers_ind = _get_joint_sub_super_stabilisers_ind(
        previous_stabilisers=previous_stabilisers,
        current_stabilisers=current_stabilisers_before,
    )

    current_offset = len(current_stabilisers_before)
    previous_offset = current_offset + len(previous_stage.ordered_stabilisers)

    # Collect detectors
    for indices_prev, indices_curr in joint_sub_super_stabilisers_ind:
        # Add indices for current stabiliser measurement
        rec_indices = [
            MeasurementRecord(ind_curr - current_offset) for ind_curr in indices_curr
        ]

        # Add indices from previous stabiliser measurement, ignoring resets
        rec_indices += [
            MeasurementRecord(ind_prev - previous_offset)
            for ind_prev in indices_prev
            if ind_prev < len(previous_stage.ordered_stabilisers)
        ]

        rec_indices.sort(key=lambda meas_rec: meas_rec.lookback_index, reverse=True)

        # Determine if "big" stabiliser is from previous or current stage
        big_stab_from_current_stage = len(indices_curr) == 1

        # Get the "big" stabiliser
        big_stabiliser = (
            current_stabilisers[indices_curr[0]]
            if big_stab_from_current_stage
            else previous_stabilisers[indices_prev[0]]
        )
        # Get the detector coordinate
        stage = current_stage if big_stab_from_current_stage else previous_stage
        stab_index = stage.ordered_stabilisers.index(big_stabiliser)
        detector_coordinate = stage.detector_coordinates[stab_index]

        detectors.append(
            Detector(
                measurements=rec_indices,
                coordinate=detector_coordinate,
            )
        )

    return detectors


def get_stage_transition_circuit(
    previous_stage: "CSSStage", current_stage: "CSSStage"
) -> Circuit:
    """
    Return a Circuit containing all the DETECTOR and SHIFT_COORD information for
    transitioning between previous_stage and current_stage stages.

    Parameters
    ----------
    previous_stage : CSSStage
        A CSSStage from which we transition. For instance, for patch growing
        previous_stage would be the stage where we measure a small patch for some
        rounds.
    current_stage : CSSStage
        The CSSStage into which we intend to transition. For instance, for patch
        growing current_stage would be the stage where we measure the bigger patch for
        some rounds.

    Returns
    -------
    Circuit
        The detectors and the coordinate shift instruction.
    """
    detectors = _get_stage_transition_detectors(previous_stage, current_stage)
    shifts = _get_coordinate_shifts(detectors)

    return Circuit(detectors + shifts)
