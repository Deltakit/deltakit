# (c) Copyright Riverlane 2020-2025.
"""This module accumulates classes for data transformation.
Contantins converters between most popular formats: 01, b8, and csv"""
from __future__ import annotations

from codecs import StreamReader, StreamWriter
from collections.abc import Iterable
from io import BufferedIOBase, BytesIO, RawIOBase, StringIO, TextIOWrapper
from pathlib import Path

import numpy.typing as npt

from deltakit_explorer.enums._api_enums import DataFormat


def read_01(stream: StreamReader | TextIOWrapper) -> Iterable[list[int]]:
    """Read 01 data from steam, StringIO or file descriptor."""
    for line in stream:
        yield [int(char) for char in line.strip()]

def read_csv(stream: StreamReader | TextIOWrapper) -> Iterable[list[int]]:
    """Read CSV data from steam, StringIO or file descriptor."""
    for line in stream:
        yield [int(char) for char in line.strip().split(",")]

def read_b8(
    stream: BufferedIOBase | RawIOBase,
    width: int
) -> Iterable[list[int]]:
    """Read bit-packed data from steam, BytesIO or file descriptor."""
    bytes_per_shot = (width + 7) // 8
    while (shot := stream.read(bytes_per_shot)):
        result = []
        # iterate through bits
        for k in range(width):
            which_byte = shot[k // 8]
            which_bit = (which_byte >> (k % 8)) % 2
            result.append(which_bit)
        yield result

def write_01(
    data: Iterable[list[int]] | npt.NDArray,
    stream: StreamWriter | TextIOWrapper | StringIO,
):
    """Write data into a text stream in 01 format."""
    for row in data:
        # convert to string, and then join with empty
        stream.write("".join(map(str, row)) + "\n")

def write_csv(
    data: Iterable[list[int]] | npt.NDArray,
    stream: StreamWriter | TextIOWrapper | StringIO,
):
    """Write data into a text stream in CSV format."""
    for row in data:
        # convert to string, and then join with empty
        stream.write(",".join(map(str, row)) + "\n")

def write_b8(
    data: Iterable[list[int]] | npt.NDArray,
    stream: BufferedIOBase | RawIOBase | BytesIO,
):
    """Write data into a bytes stream in B8 format."""
    for row in data:
        bytes_per_shot = (len(row) + 7) // 8
        longint = 0
        for bit in reversed(row):
            int_bit = int(bit)
            longint <<= 1
            longint += int_bit
        stream.write(longint.to_bytes(bytes_per_shot, "little"))

def write_binary_data_to_file(
    data: Iterable[list[int]] | npt.NDArray,
    destination_format: DataFormat,
    result_file: str | Path,
):
    """Writes a stream of bit lines (each List[int] is a line of {0, 1})
    into a file of desired format.

    Args:
        data (Union[Iterable[List[int]], npt.NDArray]):
            iterable of bit strings, or numpy array.
        destination_format (DataFormat):
            one of popular format, 01, b8, or csv.
        result_file (str | Path):
            str or Path object. Path to a file.
    """
    if destination_format == DataFormat.F01:
        with Path.open(Path(result_file), "w", encoding="utf-8") as string_stream:
            write_01(data, string_stream)

    elif destination_format == DataFormat.CSV:
        with Path.open(Path(result_file), "w", encoding="utf-8") as string_stream:
            write_csv(data, string_stream)

    elif destination_format == DataFormat.B8:
        with Path.open(Path(result_file), "wb") as byte_stream:
            write_b8(data, byte_stream)
    else:
        msg = f"Not supported format: {destination_format}"
        raise NotImplementedError(msg)
