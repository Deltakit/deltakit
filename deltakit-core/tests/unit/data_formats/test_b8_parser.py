# (c) Copyright Riverlane 2020-2025.
import pathlib
import tempfile

import pytest
import stim
from deltakit_core.data_formats import (
    b8_to_logical_flip,
    b8_to_measurements,
    b8_to_syndromes,
    logical_flips_to_b8_file,
    parse_01_to_logical_flips,
    parse_01_to_syndromes,
    syndromes_to_b8_file,
)
from deltakit_core.data_formats._b801_parsers import to_bytearray
from deltakit_core.decoding_graphs import Bitstring, OrderedSyndrome


def test_to_bytearray(reference_data_dir):
    # test that the two uses of to_bytearray are equivalent
    example_b8_file = reference_data_dir / "b801" / "detection_events.b8"
    with open(example_b8_file, "rb") as b8_data:
        res = to_bytearray(b8_data.read())
    ref = to_bytearray(example_b8_file)
    assert res == ref
    assert isinstance(res, bytearray)


class TestB8ReadWriteMethods:
    def test_example_b8_syndrome_parse(self, reference_data_dir):
        """Test the first 19 syndromes are parsed correctly from the reference
        data file.
        """
        example_b8_file = reference_data_dir / "b801" / "detection_events.b8"
        parsed_syndromes = b8_to_syndromes(example_b8_file, 40)
        target_syndromes = [
            OrderedSyndrome([4, 5, 9]),
            OrderedSyndrome([10, 27, 30, 38]),
            OrderedSyndrome([13, 21]),
            OrderedSyndrome([0, 7, 20, 21, 23]),
            OrderedSyndrome([18, 21, 22, 24, 32, 36]),
            OrderedSyndrome([32, 35]),
            OrderedSyndrome([7, 28, 29, 30, 31, 38]),
            OrderedSyndrome([8, 12, 13, 17, 26]),
            OrderedSyndrome([21, 22, 24, 27, 35]),
            OrderedSyndrome([10, 23, 25]),
            OrderedSyndrome([32, 34, 38]),
            OrderedSyndrome([13, 26, 30, 32]),
            OrderedSyndrome([10, 11, 19, 27]),
            OrderedSyndrome([]),
            OrderedSyndrome([1, 8, 9, 13, 15, 16, 18, 19, 24, 26, 35, 36, 39]),
            OrderedSyndrome([15, 18, 26, 27]),
            OrderedSyndrome([10, 20, 21, 24, 39]),
            OrderedSyndrome([1, 7, 9, 23, 25, 26, 34, 36]),
            OrderedSyndrome([2, 3, 11, 16, 27]),
        ]
        assert all(
            parsed == target
            # strict=False because only the first 19 are checked
            for parsed, target in zip(parsed_syndromes, target_syndromes, strict=False)
        )

    @pytest.mark.parametrize(
        "bits,length",
        [
            *[(0, length) for length in (2, 4, 7, 8, 9, 41)],
            *[(1 << length - 1, length) for length in (2, 4, 7, 8, 9, 41)],
            (0b1010_1111_0000, 12),
            (0b1111_1111_0000_0000_1010_0101_1110_0001_0100_1001, 40),
        ],
    )
    def test_example_b8_measurements_parse(self, bits: int, length: int):
        # test that `b8_to_measurements` returns reasonable result given example like
        # https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#the-b8-format
        bitsstr = " ".join(reversed(f"{bits:0{length}b}"))
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "tmp.dat"
            stim.Circuit(f"X 1\nM {bitsstr}").compile_sampler().sample_write(
                shots=2, filepath=str(path), format="b8"
            )
            res = list(b8_to_measurements(path, length))

        assert res == [Bitstring(bits), Bitstring(bits)]

    @pytest.mark.parametrize(
        "file_name, expected_logical_flips, num_logicals",
        [
            (
                "obs_flips_one_logical.b8",
                [
                    (False,),
                    (False,),
                    (False,),
                    (False,),
                    (True,),
                    (False,),
                    (True,),
                    (True,),
                    (False,),
                    (True,),
                    (False,),
                    (False,),
                    (False,),
                    (False,),
                    (True,),
                    (False,),
                    (True,),
                    (True,),
                    (False,),
                    (False,),
                ],
                1,
            ),
            (
                "obs_flips_two_logical.b8",
                [
                    (False, False),
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                    (False, True),
                    (True, True),
                    (True, False),
                    (False, False),
                    (True, True),
                ],
                2,
            ),
        ],
    )
    def test_example_b8_logical_flip_parse(
        self, reference_data_dir, file_name, expected_logical_flips, num_logicals
    ):
        """Test the first few logical flips are parsed correctly from the reference
        data files.
        """
        example_b8_file = reference_data_dir / "b801" / file_name
        parsed_logical_flips = b8_to_logical_flip(example_b8_file, num_logicals)
        assert all(
            parsed == target
            for parsed, target in zip(
                parsed_logical_flips, expected_logical_flips, strict=True
            )
        )

    def test_example_01_logical_flip_parse(self, reference_data_dir):
        example_01_file = reference_data_dir / "b801" / "logical_flips.01"
        parsed_logical_flips = parse_01_to_logical_flips(example_01_file)
        expected_logical_flips = [
            (False, True),
            (False, False),
            (True, True),
            (True, False),
            (True, False),
        ]
        assert all(
            parsed == target
            for parsed, target in zip(
                parsed_logical_flips, expected_logical_flips, strict=True
            )
        )

    def test_example_01_syndrome_parse(self, reference_data_dir):
        example_01_file = reference_data_dir / "b801" / "syndromes.01"
        parsed_logical_flips = parse_01_to_syndromes(example_01_file)
        expected_syndromes = [OrderedSyndrome([1, 5, 6]), OrderedSyndrome([2, 3])]
        assert all(
            parsed == target
            for parsed, target in zip(
                parsed_logical_flips, expected_syndromes, strict=True
            )
        )

    def test_example_b8_logical_flip_read_write_equivalence(
        self, reference_data_dir, tmp_path
    ):
        example_b8_file = (
            reference_data_dir / "b801" / "obs_flips_predicted_by_pymatching.b8"
        )
        parsed_syndromes = b8_to_logical_flip(example_b8_file, 1)
        tmp_b8_out_file = tmp_path / "tmp_logical_flips_out.b8"
        logical_flips_to_b8_file(tmp_b8_out_file, parsed_syndromes)
        with open(example_b8_file, "rb") as origin_handle:
            with open(tmp_b8_out_file, "rb") as out_handle:
                assert origin_handle.read() == out_handle.read()

    def test_example_b8_syndrome_read_write_equivalence(
        self, reference_data_dir, tmp_path
    ):
        example_b8_file = reference_data_dir / "b801" / "detection_events.b8"
        parsed_syndromes = b8_to_syndromes(example_b8_file, 40)
        tmp_b8_out_file = tmp_path / "tmp_detection_events_out.b8"
        syndromes_to_b8_file(tmp_b8_out_file, 40, parsed_syndromes)
        with open(example_b8_file, "rb") as origin_handle:
            with open(tmp_b8_out_file, "rb") as out_handle:
                assert origin_handle.read() == out_handle.read()
