# (c) Copyright Riverlane 2020-2025.
import csv
from collections.abc import Iterable, Iterator
from itertools import cycle
from pathlib import Path

from deltakit_core.decoding_graphs import Bitstring


def split_input_data_to_c64(
    input_data: Iterable[Iterable[Bitstring]], output_path: Path
):
    """
    Given a list of round-split decoder input data, this will output them to a file
    as comma separated 64-bit integers. Each line is a new shot.

    The decoder input data could be measurements or syndromes.
    """
    with output_path.open("w", newline="", encoding="ascii") as output_handle:
        cw = csv.writer(output_handle)
        for input_datum in input_data:
            words = [
                int(item)
                for line in [
                    input_datum_round.to_words(64) for input_datum_round in input_datum
                ]
                for item in line
            ]
            cw.writerow(words)


def c64_to_addressed_input_words(
    input_path: Path, round_width: int
) -> Iterator[list[tuple[int, int]]]:
    """Given a path to a c64 file, return, for each line, the words within the line
    and the index of each word in each round. These are appropriate for input to a
    hardware decoder, they could be syndromes or measurement data.

    Parameters
    ----------
    input_path : Path
        Path to the c64 file to read.
    round_width : int
        The number of detectors or measurements per round.

    Yields
    ------
    List[Tuple[int, int]]
        List of addressed 64bit words
    """
    with input_path.open("r", encoding="ascii") as input_handle:
        syndrome_reader = csv.reader(input_handle)
        for row in syndrome_reader:
            round_indices = cycle(range(round_width // 64 + 1))
            yield [
                (word_index, int(word_64))
                for word_index, word_64 in zip(round_indices, row, strict=False)
            ]
