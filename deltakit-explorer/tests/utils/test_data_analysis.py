# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from pathlib import Path

import pytest
import stim

from deltakit_explorer.data._data_analysis import (
    get_binary_data_size,
    get_decoding_request_size_estimate,
    get_decoding_response_size_estimate,
    get_simulation_response_size_estimate,
)
from deltakit_explorer.enums import DataFormat


@pytest.mark.parametrize(
    ("width", "shots", "data_format", "expected"),
    [
        (1, 1, DataFormat.F01, 2),  # +endline
        (14, 1, DataFormat.F01, 15),
        (1, 100, DataFormat.F01, 200),  # 100 lines, 2 bytes each
        (14, 100, DataFormat.F01, 1500),  # 100 lines, 15 bytes each
        (1, 1, DataFormat.CSV, 2),
        (14, 1, DataFormat.CSV, 28),
        (1, 100, DataFormat.CSV, 200),  # 100 lines, 2 bytes each
        (14, 100, DataFormat.CSV, 2800),  # 100 lines, 28 bytes each
        (1, 1, DataFormat.B8, 1),
        (14, 1, DataFormat.B8, 2),
        (1, 100, DataFormat.B8, 100),  # 100 lines, 1 bytes each
        (14, 100, DataFormat.B8, 200),  # 100 lines, 2 bytes each
    ],
)
def test_binary_data_size(
    width: int, shots: int, data_format: DataFormat, expected: int
):
    assert get_binary_data_size(width, shots, data_format) == expected


def test_binary_data_size_raises():
    with pytest.raises(NotImplementedError):
        get_binary_data_size(100, 100, DataFormat.TEXT)


@pytest.mark.parametrize(
    ("shots", "data_format", "expected"),
    [
        (1, DataFormat.F01, 238),
        (100, DataFormat.F01, 2614),
        (1, DataFormat.CSV, 250),
        (100, DataFormat.CSV, 3814),
        (1, DataFormat.B8, 218),
        (100, DataFormat.B8, 614),
    ],
)
def test_get_decoding_request_size_estimate(
    shots: int, data_format: DataFormat, expected: int
):
    stim_file = Path(__file__).parent / "../resources/rep_code_mutated_default_noise_data.stim"
    circuit = stim.Circuit.from_file(stim_file)
    circuit_text = str(circuit)
    assert get_decoding_request_size_estimate(circuit, shots, data_format) == expected
    assert (
        get_decoding_request_size_estimate(circuit_text, shots, data_format) == expected
    )


@pytest.mark.parametrize(
    ("shots", "data_format", "expected"),
    [
        (1, DataFormat.F01, 240),
        (100, DataFormat.F01, 2814),
        (1, DataFormat.CSV, 254),
        (100, DataFormat.CSV, 4214),
        (1, DataFormat.B8, 218),
        (100, DataFormat.B8, 614),
    ],
)
def test_get_decoding_request_size_estimate_with_leakage(
    shots: int, data_format: DataFormat, expected: int
):
    circuit_text = """
    M 0 1 2 3 4
    M 5 6 7 8
    HERALD_LEAKAGE_EVENT(0.1) 1 2 3 4
    OBSERVABLE_INCLUDE
    OBSERVABLE_INCLUDE
    OBSERVABLE_INCLUDE
    HERALD_LEAKAGE_EVENT(0.1) 5 6 7
    """
    assert (
        get_decoding_request_size_estimate(circuit_text, shots, data_format) == expected
    )


@pytest.mark.parametrize(
    ("shots", "data_format", "expected"),
    [
        (1, DataFormat.F01, 211),
        (100, DataFormat.F01, 607),
        (1, DataFormat.CSV, 211),
        (100, DataFormat.CSV, 607),
        (1, DataFormat.B8, 209),
        (100, DataFormat.B8, 407),
    ],
)
def test_get_decoding_response_size_estimate(
    shots: int, data_format: DataFormat, expected: int
):
    stim_file = Path(__file__).parent / "../resources/rep_code_mutated_default_noise_data.stim"
    circuit = stim.Circuit.from_file(stim_file)
    circuit_text = str(circuit)
    assert get_decoding_response_size_estimate(circuit, shots, data_format) == expected
    assert (
        get_decoding_response_size_estimate(circuit_text, shots, data_format)
        == expected
    )


@pytest.mark.parametrize(
    ("shots", "data_format", "expected"),
    [
        (1, DataFormat.F01, 236),
        (100, DataFormat.F01, 2414),
        (1, DataFormat.CSV, 250),
        (100, DataFormat.CSV, 3814),
        (1, DataFormat.B8, 218),
        (100, DataFormat.B8, 614),
    ],
)
def test_get_simulation_response_size_estimate(
    shots: int, data_format: DataFormat, expected: int
):
    stim_file = Path(__file__).parent / "../resources/rep_code_mutated_default_noise_data.stim"
    circuit = stim.Circuit.from_file(stim_file)
    circuit_text = str(circuit)
    assert (
        get_simulation_response_size_estimate(circuit, shots, data_format) == expected
    )
    assert (
        get_simulation_response_size_estimate(circuit_text, shots, data_format)
        == expected
    )


@pytest.mark.parametrize(
    ("shots", "data_format", "expected"),
    [
        (1, DataFormat.F01, 236),
        (100, DataFormat.F01, 2414),
        (1, DataFormat.CSV, 250),
        (100, DataFormat.CSV, 3814),
        (1, DataFormat.B8, 218),
        (100, DataFormat.B8, 614),
    ],
)
def test_simulation_response_size_estimate_with_leakage(
    shots: int, data_format: DataFormat, expected: int
):
    circuit_text = """
    M 0 1 2 3 4
    M 5 6 7 8
    HERALD_LEAKAGE(0.1) 1 2 3 4
    OBSERVABLE_INCLUDE
    OBSERVABLE_INCLUDE
    OBSERVABLE_INCLUDE
    OBSERVABLE_INCLUDE
    HERALD_LEAKAGE(0.1) 5 6 7
    """
    assert (
        get_simulation_response_size_estimate(circuit_text, shots, data_format)
        == expected
    )

def test_get_binary_data_size_b8():
    assert get_binary_data_size(8, 2, DataFormat.B8) == 2

def test_get_binary_data_size_csv():
    assert get_binary_data_size(3, 2, DataFormat.CSV) == 12

def test_get_binary_data_size_invalid():
    with pytest.raises(NotImplementedError):
        get_binary_data_size(1, 1, "INVALID")
