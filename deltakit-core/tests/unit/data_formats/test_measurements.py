# (c) Copyright Riverlane 2020-2025.

from pathlib import Path

import pytest

from deltakit_core.data_formats._measurements import (
    c64_to_addressed_input_words,
    split_input_data_to_c64,
)
from deltakit_core.decoding_graphs import Bitstring


class TestMeasurements:
    @pytest.mark.parametrize(
        ("bitstrings", "c64_string"),
        [
            ([[Bitstring(0b10000010)]], "130\n"),
            ([[Bitstring(0b10000010), Bitstring(0b111000111)]], "130,455\n"),
            (
                [
                    [Bitstring(0b10000010), Bitstring(0b111000111)],
                    [Bitstring(0b00110111)],
                ],
                "130,455\n55\n",
            ),
            ([[Bitstring.from_indices([65, 66, 67])]], "0,14\n"),
            ([[Bitstring.from_indices([20, 63, 65])]], "9223372036855824384,2\n"),
        ],
    )
    def test_split_input_data_to_c64_gives_correct_results(
        self, tmp_path, bitstrings, c64_string
    ) -> None:
        tmp_c64_file: Path = tmp_path / "tmp_split_measurement_events_out.c64"
        split_input_data_to_c64(bitstrings, tmp_c64_file)
        with tmp_c64_file.open("r") as c64_handle:
            assert c64_handle.read() == c64_string

    @pytest.mark.parametrize(
        ("c64_lines", "round_width", "expected_addressed_words"),
        [
            (
                "0, 14, 13, 4, 1\n4",
                4,
                [[(0, 0), (0, 14), (0, 13), (0, 4), (0, 1)], [(0, 4)]],
            ),
            (
                "128121, 1111, 0, 181, 4, 52391, 8",
                150,
                [
                    [
                        (0, 128121),
                        (1, 1111),
                        (2, 0),
                        (0, 181),
                        (1, 4),
                        (2, 52391),
                        (0, 8),
                    ]
                ],
            ),
            (  # Example of input with balanced words per round
                "24414, 1325135, 2153135, 67895657",
                100,
                [[(0, 24414), (1, 1325135), (0, 2153135), (1, 67895657)]],
            ),
            ("", 3, []),
        ],
    )
    def test_c64_to_addressed_input_words(
        self, tmp_path, c64_lines, round_width, expected_addressed_words
    ):
        tmp_c64_file: Path = tmp_path / "tmp_addressed_words.c64"
        with tmp_c64_file.open("w") as c64_handle:
            c64_handle.write(c64_lines)
        assert (
            list(c64_to_addressed_input_words(tmp_c64_file, round_width))
            == expected_addressed_words
        )
