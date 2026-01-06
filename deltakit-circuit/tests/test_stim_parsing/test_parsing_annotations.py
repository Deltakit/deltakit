# (c) Copyright Riverlane 2020-2025.
import pytest
import stim

import deltakit_circuit as sp


@pytest.mark.parametrize(
    ("stim_circuit", "expected_annotation"),
    [
        (stim.Circuit("DETECTOR rec[-1]"), sp.Detector(sp.MeasurementRecord(-1))),
        (
            stim.Circuit("DETECTOR rec[-1] rec[-2]"),
            sp.Detector([sp.MeasurementRecord(-1), sp.MeasurementRecord(-2)]),
        ),
        (
            stim.Circuit("DETECTOR(0, 1, 2) rec[-1]"),
            sp.Detector(sp.MeasurementRecord(-1), sp.Coordinate(0, 1, 2)),
        ),
        (
            stim.Circuit("DETECTOR(1, 2) rec[-1] rec[-2]"),
            sp.Detector(
                [sp.MeasurementRecord(-1), sp.MeasurementRecord(-2)],
                sp.Coordinate(1, 2),
            ),
        ),
        (
            stim.Circuit("X 0 1\nCNOT 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]"),
            sp.Detector([sp.MeasurementRecord(-1), sp.MeasurementRecord(-2)]),
        ),
        (
            stim.Circuit("OBSERVABLE_INCLUDE(0) rec[-1]"),
            sp.Observable(0, sp.MeasurementRecord(-1)),
        ),
        (stim.Circuit("SHIFT_COORDS(0, 0, 1)"), sp.ShiftCoordinates((0, 0, 1))),
    ],
)
def test_parsing_stim_circuit_with_single_annotation_puts_expected_annotation_into_deltakit_circuit_circuit_layers(
    stim_circuit: stim.Circuit, expected_annotation
):
    assert expected_annotation in sp.Circuit.from_stim_circuit(stim_circuit).layers
