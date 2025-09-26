# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from deltakit_core.decoding_graphs import Bitstring, OrderedSyndrome


def to_bytearray(data: Path | bytes) -> bytearray:
    """Converts either the content of a file or a bytes object to
    a bytearray object. A key motivation for this is that np.unpackbits
    accepts bytearray but not bytes."""
    if isinstance(data, Path):
        with open(data, "rb") as b8_data:
            return bytearray(b8_data.read())
    return bytearray(data)


def b8_to_syndromes(
    b8_input: Path | bytes, detector_num: int
) -> Iterator[OrderedSyndrome]:
    """Given a b8 input (either a file path containing b8 data or bytes),
    and the number of detectors that should be in each syndrome, return a
    generator of syndromes.

    Informed by https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#b8
    """
    bytes_per_shot = (detector_num + 7) // 8
    b8_byte_arr: bytearray = to_bytearray(b8_input)
    for offset in range(0, len(b8_byte_arr), bytes_per_shot):
        yield OrderedSyndrome.from_bitstring(
            np.unpackbits(
                np.frombuffer(
                    b8_byte_arr[offset : offset + bytes_per_shot], dtype=np.uint8
                ),
                bitorder="little",
            ).tolist()
        )


def b8_to_measurements(b8_input: Path, measurement_num: int) -> Iterator[Bitstring]:
    """Given a b8 input (a file path containing b8 data),
    and the number of measurements in each shot,
    return a generator of measurement bitstrings.

    Informed by https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#b8
    """
    bytes_per_shot = (measurement_num + 7) // 8
    b8_byte_arr: bytearray = to_bytearray(b8_input)
    for offset in range(0, len(b8_byte_arr), bytes_per_shot):
        yield Bitstring.from_bits(
            np.unpackbits(
                np.frombuffer(
                    b8_byte_arr[offset : offset + bytes_per_shot], dtype=np.uint8
                ),
                bitorder="little",
            ).tolist()
        )


def b8_to_logical_flip(
    b8_input: Path | bytes, num_logicals: int = 1
) -> Iterator[Tuple[bool, ...]]:
    """Given a b8 input (either a file path containing b8 data or bytes),
    and the number of detectors that should be in each syndrome, return a
    generator of logical flips.

    Informed by https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#b8
    """
    bytes_per_shot = (num_logicals + 7) // 8
    b8_byte_arr: bytearray = to_bytearray(b8_input)
    for offset in range(0, len(b8_byte_arr), bytes_per_shot):
        yield tuple(
            np.unpackbits(
                np.frombuffer(
                    b8_byte_arr[offset : offset + bytes_per_shot], dtype=np.uint8
                ),
                bitorder="little",
            )[:num_logicals].tolist(),
        )


def syndromes_to_b8_file(
    syndrome_b8_out: Path, detector_num: int, syndromes: Iterator[OrderedSyndrome]
):
    """Given a syndrome generator, this will output them to file in the b8 format.
    Informed by https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#b8
    """
    with open(syndrome_b8_out, "wb") as b8_out_handle:
        for syndrome in syndromes:
            b8_out_handle.write(
                bytes(
                    np.packbits(syndrome.as_bitstring(detector_num), bitorder="little")
                )
            )


def logical_flips_to_b8_file(
    logical_flips_b8_out: Path, logical_flips: Iterator[Tuple[bool, ...]]
):
    """Given a logical flip generator, this will output them to file in the b8 format.
    b8 is a byte-padded bitstring representation for logical flips. This is more
    efficient than the string-based 01 representation.
    Informed by https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#b8
    """
    with open(logical_flips_b8_out, "wb") as data_b8:
        for logical_flip in logical_flips:
            data_b8.write(bytes(np.packbits(logical_flip, bitorder="little")))


def parse_01_to_logical_flips(
    logical_flips_01_file: Path,
) -> Iterator[Tuple[bool, ...]]:
    """Given a filepath for a 01 file containing target logical flips, return a
    generator of logical flips as booleans.

    01 files are string based, where each line is a different shot. Where 1
    indicates the reference logical is flipped.

    Use logical_flips_to_b8_file on the result of this function to translate 01
    data to b8 data that can be decoded via the B8DecoderManager.
    """
    with open(logical_flips_01_file, "r", encoding="utf-8") as logical_flips_handle:
        for bit_string in logical_flips_handle.read().splitlines():
            yield tuple(char != "0" for char in bit_string)


def parse_01_to_syndromes(
    logical_flips_01_file: Path,
) -> Iterator[OrderedSyndrome]:
    """Given a filepath for a 01 file containing syndromes, return a
    generator of OrderedSyndromes.

    01 files are string based, where each line is a different target logical as a
    string of 0s and 1s. Where 1 indicates the reference logical is flipped.

    Use logical_flips_to_b8_file on the result of this function to translate 01
    data to b8 data that can be decoded via the B8DecoderManager.
    """
    with open(logical_flips_01_file, "r", encoding="utf-8") as logical_flips_handle:
        for bit_string in logical_flips_handle.read().splitlines():
            yield OrderedSyndrome([i for i, v in enumerate(bit_string) if v == "1"])
