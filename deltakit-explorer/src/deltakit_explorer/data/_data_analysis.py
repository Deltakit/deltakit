# (c) Copyright Riverlane 2020-2025.
"""Module contains functions to perform analysis of data assets to perform efficient
data routing between client and service."""
from __future__ import annotations

import re

from stim import Circuit

from deltakit_explorer.enums._api_enums import DataFormat
from deltakit_explorer.types._data_string import DataString


def get_binary_data_size(width: int, shots: int, data_format: DataFormat) -> int:
    """Get number of bytes to represent a binary table (width x shots)
    using the data_format.

    Args:
       width (int): bits per shot.
       shots (int): number of shots.
       data_format (DataFormat): how the data is packed.

    Returns:
        int: size in memory, in bytes
    """
    if data_format == DataFormat.B8:
        # each shot is a bit
        length = (width + 7) // 8
    elif data_format == DataFormat.F01:
        # + endline
        length = width + 1
    elif data_format == DataFormat.CSV:
        # commas + endline
        length = 2 * width
    else:
        msg = f"Format {data_format} is not supported for binary data"
        raise NotImplementedError(
            msg
        )
    return length * shots


def get_decoding_response_size_estimate(
    circuit: str | Circuit,
    shots: int,
    data_format: DataFormat,
) -> int:
    """Get an estimation for a decoding response, in bytes.
    Size may slightly vary based on moving parts
    of the request. Estimation is an upper bound.

    Args:
       circuit (Union[str, Circuit]): a circuit, for which detectors are decoded.
       shots (int): number of shots to decode.
       data_format (DataFormat): how the data is packed.

    Returns:
        int: size of response content, in bytes
    """
    if isinstance(circuit, Circuit):
        width = circuit.num_observables
    else:
        width = len(re.findall("OBSERVABLE_INCLUDE", circuit))
    bytes_data_size = get_binary_data_size(width, shots, data_format)
    duck_size = len(DataString.empty) + 2 * bytes_data_size
    # all other field, including shots, errors, times, quotes
    # empirical upper bound estimate
    wrapper_size = 200
    return wrapper_size + duck_size


def get_decoding_request_size_estimate(
    circuit: str | Circuit,
    shots: int,
    data_format: DataFormat,
) -> int:
    """Get an estimation for a decoding request, in bytes.
    Size may slightly vary based on moving parts
    of the request. Estimation is an upper bound.

    Args:
       circuit (Union[str, Circuit]): a circuit, for which detectors are decoded.
       shots (int): number of shots to decode.
       data_format (DataFormat): how the data is packed.

    Returns:
        int: size of request content, in bytes
    """
    if isinstance(circuit, Circuit):
        width_det = circuit.num_detectors
        width_obs = circuit.num_observables
        width_leakage = 0
    else:
        width_det = len(re.findall(r"DETECTOR\(", circuit))
        width_obs = len(re.findall(r"OBSERVABLE_INCLUDE", circuit))
        width_leakage = 0
        for line in circuit.splitlines(False):
            line = line.strip()
            # add all measurements and leakages
            if line.startswith("HERALD_LEAKAGE_EVENT"):
                # there is 1 operator followed by arguments
                width_leakage += len(line.split()) - 1
    duck_size = 2 * (
        len(DataString.empty)  # used twice
        + get_binary_data_size(width_det, shots, data_format)
        + get_binary_data_size(width_obs, shots, data_format)
        + get_binary_data_size(width_leakage, shots, data_format)
    )
    # all other field, including variable names, formats, ...
    # empirical upper bound estimate
    wrapper_size = 200
    return wrapper_size + duck_size


def get_simulation_response_size_estimate(
    circuit: str | Circuit,
    shots: int,
    data_format: DataFormat,
) -> int:
    """Get an estimation for a simulation response, in bytes.
    Size may slightly vary based on moving parts
    of the response. Estimation is an upper bound.

    Args:
       circuit (Union[str, Circuit]): a circuit to simulate.
       shots (int): number of shots to simulate.
       data_format (DataFormat): how the data is packed.

    Returns:
        int: size of response content, in bytes
    """
    if isinstance(circuit, Circuit):
        width_mmt = circuit.num_measurements
        width_leak = 0
    else:
        width_mmt, width_leak = 0, 0
        for line in circuit.splitlines(False):
            line = line.strip()
            # add all measurements and leakages
            if line.startswith("M"):
                width_mmt += len(line.split()) - 1
            elif line.startswith("HERALD_LEAKAGE_EVENT"):
                # there is 1 operator followed by arguments
                width_leak += len(line.split()) - 1
    duck_size = 2 * (
        len(DataString.empty)
        + get_binary_data_size(width_mmt, shots, data_format)
        + get_binary_data_size(width_leak, shots, data_format)
    )
    # all other field, including formats, quotes, ...
    # empirical upper bound estimate
    wrapper_size = 200
    return wrapper_size + duck_size


def has_leakage(stim_circuit: str) -> bool:
    """Check is circuit has any leakage-related instructions.
    If it has, if should only be stored in strings and processed
    as client.

    Args:
        stim_circuit (str): string representation of STIM circuit.

    Returns:
        bool: True if any leakage-related instruction are there.
    """
    leakage_evidence = {"LEAKAGE", "HERALD_LEAKAGE_EVENT", "RELAX"}
    stim_text = str(stim_circuit)
    stim_lines = stim_text.splitlines()
    return any(
        line.strip().startswith(prefix)
        for line in stim_lines
        for prefix in leakage_evidence
    )
