# (c) Copyright Riverlane 2020-2025.
import pytest
import stim
from deltakit_circuit import MeasurementRecord, Observable


def test_error_is_raised_if_the_observable_index_is_less_than_zero():
    with pytest.raises(ValueError, match="Observable index cannot be negative."):
        Observable(-1, MeasurementRecord(-1))


def test_observable_measurements_only_contain_unique_measurement_records():
    observable = Observable(1, (MeasurementRecord(-1), MeasurementRecord(-1)))
    assert len(observable.measurements) == 1


@pytest.mark.parametrize(
    "observable, expected_circuit",
    [
        (
            Observable(0, MeasurementRecord(-1)),
            stim.Circuit("OBSERVABLE_INCLUDE(0) rec[-1]"),
        ),
        (
            Observable(4, MeasurementRecord(-3)),
            stim.Circuit("OBSERVABLE_INCLUDE(4) rec[-3]"),
        ),
    ],
)
def test_observable_stim_circuit_returns_expected_circuit(
    observable, expected_circuit, empty_circuit
):
    observable.permute_stim_circuit(empty_circuit)
    assert empty_circuit == expected_circuit


@pytest.mark.parametrize(
    "observable, expected_repr",
    [
        (
            Observable(0, MeasurementRecord(-1)),
            "Observable([MeasurementRecord(-1)], index=0)",
        ),
        (
            Observable(20, [MeasurementRecord(-4), MeasurementRecord(-2)]),
            "Observable([MeasurementRecord(-4), MeasurementRecord(-2)], index=20)",
        ),
    ],
)
def test_repr_of_observable_matches_expected_representation(observable, expected_repr):
    assert repr(observable) == expected_repr


@pytest.mark.parametrize(
    "observable",
    [
        Observable(3, (MeasurementRecord(-1), MeasurementRecord(-2))),
        Observable(
            2, (MeasurementRecord(-1), MeasurementRecord(-1), MeasurementRecord(-4))
        ),
    ],
)
def test_all_measurement_record_indices_are_in_observable_stim_circuit(
    observable: Observable, empty_circuit
):
    observable.permute_stim_circuit(empty_circuit)
    circuit_string = str(empty_circuit)
    assert all(
        f"rec[{record.lookback_index}]" in circuit_string
        for record in observable.measurements
    )


@pytest.mark.parametrize(
    "observable,index",
    (
        [Observable(0, MeasurementRecord(-1)), 0],
        [Observable(33, MeasurementRecord(-1)), 33],
    ),
)
def test_observable_index_read(observable: Observable, index: int, empty_circuit):
    observable.permute_stim_circuit(empty_circuit)
    assert observable.observable_index == index


def test_observable_index_write_protected():
    observable = Observable(0, MeasurementRecord(-1))
    with pytest.raises(AttributeError):
        observable.observable_index = 4


@pytest.mark.parametrize("tag", [None, "", "sjkdhf", "λ", "leaky<0>"])
def test_tag_on_instruction(tag: str | None) -> None:
    obs = Observable(0, [MeasurementRecord(-1), MeasurementRecord(-2)], tag=tag)
    assert obs.tag == tag
