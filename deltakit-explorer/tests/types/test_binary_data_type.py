# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import numpy as np
import pytest
import stim

from deltakit_explorer.enums import DataFormat
from deltakit_explorer.types import (
    BinaryDataType,
    DataString,
    DetectionEvents,
    LeakageFlags,
    Measurements,
    ObservableFlips,
)


@pytest.mark.parametrize(
    ("raw", "data_format"),
    [
        (DataString("0101\n1001"), DataFormat.F01),
        (DataString("0,1,0,1\n1,0,0,1"), DataFormat.CSV),
        (DataString(b"\x0a\x09"), DataFormat.B8),
        ([[0, 1, 0, 1], [1, 0, 0, 1]], DataFormat.F01),
        (np.array([[0, 1, 0, 1], [1, 0, 0, 1]]), DataFormat.F01),
        ([[0, 1, 0, 1], [1, 0, 0, 1]], DataFormat.B8),
        (np.array([[0, 1, 0, 1], [1, 0, 0, 1]]), DataFormat.B8),

    ],
)
def test_construct_and_convert_numpy(raw, data_format):
    obj = BinaryDataType(raw, data_format, 4)
    assert np.allclose(
        obj.as_numpy(),
        [[0, 1, 0, 1], [1, 0, 0, 1]],
    )


@pytest.mark.parametrize(
    ("raw", "data_format"),
    [
        (DataString("0101\n1001"), DataFormat.F01),
        (DataString("0,1,0,1\n1,0,0,1"), DataFormat.CSV),
        (DataString(b"\x0a\x09"), DataFormat.B8),
        ([[0, 1, 0, 1], [1, 0, 0, 1]], DataFormat.F01),
        (np.array([[0, 1, 0, 1], [1, 0, 0, 1]]), DataFormat.F01),
        ([[0, 1, 0, 1], [1, 0, 0, 1]], DataFormat.B8),
        (np.array([[0, 1, 0, 1], [1, 0, 0, 1]]), DataFormat.B8),

    ],
)
@pytest.mark.parametrize("cls", [BinaryDataType, Measurements, LeakageFlags, DetectionEvents, ObservableFlips])
@pytest.mark.parametrize("target_format", [DataFormat.F01, DataFormat.B8, DataFormat.CSV])
def test_serialise_deserialise_datastring(raw, data_format, cls, target_format):
    obj = cls(data=raw, data_format=data_format, data_width=4)
    assert np.allclose(
        obj.as_numpy(),
        [[0, 1, 0, 1], [1, 0, 0, 1]],
    )
    duckstring = obj.as_data_string(target_format)
    new_obj = cls(
        data=DataString.from_data_string(duckstring),
        data_format=target_format,
        data_width=obj.data.data_width,
    )
    assert obj == new_obj


@pytest.mark.parametrize(
    ("raw", "data_format"),
    [
        (DataString("0101\n1010"), DataFormat.F01),
        (DataString("0,1,0,1\n1,0,1,0"), DataFormat.CSV),
        (DataString(b"\x0a\x05"), DataFormat.B8),
        ([[0, 1, 0, 1], [1, 0, 1, 0]], DataFormat.F01),
        ([[0, 1, 0, 1], [1, 0, 1, 0]], DataFormat.B8),
        (np.array([[0, 1, 0, 1], [1, 0, 1, 0]]), DataFormat.F01),
        (np.array([[0, 1, 0, 1], [1, 0, 1, 0]]), DataFormat.B8),
        (np.array([[0, 1, 0, 1], [1, 0, 1, 0]]), DataFormat.CSV),
    ],
)
@pytest.mark.parametrize("cls", [BinaryDataType, Measurements, DetectionEvents, ObservableFlips, LeakageFlags])
def test_construct_and_convert_01(raw, data_format, cls):
    obj = cls(data=raw, data_format=data_format, data_width=4)
    assert obj.as_01_string().strip() == "0101\n1010"


@pytest.mark.parametrize(
    "cls", [BinaryDataType, ObservableFlips, DetectionEvents, Measurements, LeakageFlags]
)
@pytest.mark.parametrize(
    "data_format", [DataFormat.B8, DataFormat.CSV, DataFormat.F01]
)
@pytest.mark.parametrize("times", [1, 2, 3000])
def test_eq_ok(cls, data_format, times):
    assert cls([[0, 1] * times], data_format) == cls(DataString("01" * times), DataFormat.F01)


@pytest.mark.parametrize(
    "cls", [BinaryDataType, ObservableFlips, DetectionEvents, Measurements, LeakageFlags]
)
def test_eq_not_eq(cls):
    assert cls([[1, 1]], DataFormat.B8, 2) != cls(DataString("01"), DataFormat.F01, 2)


@pytest.mark.parametrize(
    "cls", [BinaryDataType, ObservableFlips, DetectionEvents, Measurements, LeakageFlags]
)
def test_eq_type_is_wrong(cls):
    assert cls([[1, 1]], DataFormat.B8, 2) != "abcd"


@pytest.mark.parametrize(
    ("cls1", "cls2"), [(BinaryDataType, ObservableFlips), (DetectionEvents, Measurements)]
)
def test_eq_type_mismatch(cls1, cls2):
    assert cls1([[0, 1]], DataFormat.B8, 2) != cls2(
        DataString("01"), DataFormat.F01,
    )


@pytest.mark.parametrize(
    "cls",
    [BinaryDataType, Measurements, DetectionEvents, ObservableFlips, LeakageFlags])
@pytest.mark.parametrize("items", [1, 2, 3])
def test_join_works(cls, items):
    data = [[0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]]
    result = []
    for i in range(items):
        result.append(cls(data[i:i+1], DataFormat.B8))
    joined = cls.combine(result)
    assert np.allclose(joined.data.content, data[:items])


@pytest.mark.parametrize(
    "cls",
    [BinaryDataType, Measurements, DetectionEvents, ObservableFlips, LeakageFlags])
@pytest.mark.parametrize("items", [0])
def test_join_raises_on_empty(cls, items):
    data = [[0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]]

    result = []
    for i in range(items):
        result.append(cls(data[i:i+1], DataFormat.B8))
    with pytest.raises(ValueError):
        cls.combine(result)


@pytest.mark.parametrize(
    "cls",
    [BinaryDataType, Measurements, DetectionEvents, ObservableFlips, LeakageFlags])
@pytest.mark.parametrize("items", [2, 3])
def test_join_raises_on_nonaligned_rows(cls, items):
    data = [[0, 1, 0, 1, 0], [0, 1, 1], [1, 0, 1, 0]]

    result = []
    for i in range(items):
        result.append(cls(data[i:i+1], DataFormat.B8))
    with pytest.raises(ValueError):
        cls.combine(result)


def test_measurements_split_detectors_observables_stim_circuit():
    mmts = Measurements([[0, 1], [0, 1]])
    circ = """
        M 0 1
        DETECTOR(1, 2) rec[-1] rec[-2]
        DETECTOR(1, 3) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    dets, obs = mmts.to_detectors_and_observables(stim.Circuit(circ))
    assert dets.as_numpy().shape == (2, 2)
    assert obs.as_numpy().shape == (2, 1)

def test_measurements_split_detectors_observables_stim():
    mmts = Measurements([[0, 1], [0, 1]])
    circ = """
        M 0 1
        DETECTOR(1, 2) rec[-1] rec[-2]
        DETECTOR(1, 3) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    dets, obs = mmts.to_detectors_and_observables(circ)
    assert dets.as_numpy().shape == (2, 2)
    assert obs.as_numpy().shape == (2, 1)


def test_measurements_split_detectors_observables_stim_with_leakage():
    mmts = Measurements([[0, 1, 0, 0], [0, 0, 0, 1]])
    circ = """
        M 0 1
        DETECTOR(1, 2) rec[-1] rec[-2]
        DETECTOR(1, 3) rec[-1]
        HERALD_LEAKAGE_EVENT 0
        M 0 1
        DETECTOR(1, 2) rec[-1] rec[-2]
        DETECTOR(1, 3) rec[-1]
        HERALD_LEAKAGE_EVENT 0
        OBSERVABLE_INCLUDE(0) rec[-2]
        DETECTOR rec[-1]
        """
    dets, obs = mmts.to_detectors_and_observables(circ)
    assert dets.as_numpy().shape == (2, 4)
    assert obs.as_numpy().shape == (2, 1)


def test_measurements_split_detectors_observables_stim_with_sweeps():
    mmts = Measurements([[0, 1, 0, 0], [0, 0, 0, 1]])
    circ = """
        CX sweep[0] 0
        M 0 1
        DETECTOR(1, 2) rec[-1] rec[-2]
        DETECTOR(1, 3) rec[-1]
        HERALD_LEAKAGE_EVENT 0
        M 0 1
        RELAX 0
        DETECTOR(1, 2) rec[-1] rec[-2]
        DETECTOR(1, 3) rec[-1]
        HERALD_LEAKAGE_EVENT 0
        OBSERVABLE_INCLUDE(0) rec[-2]
        """
    dets, obs = mmts.to_detectors_and_observables(circ, BinaryDataType([[0], [0]]))
    assert dets.as_numpy().shape == (2, 4)
    assert obs.as_numpy().shape == (2, 1)


def test_measurements_split_detectors_observables_bad_circuit():
    mmts = Measurements([[0, 1, 0, 0], [0, 0, 0, 1]])
    circ = """BAD ONE"""
    with pytest.raises(ValueError):
        mmts.to_detectors_and_observables(circ, BinaryDataType([[0], [0]]))
