# (c) Copyright Riverlane 2020-2025.
"""`Simulation` module aggregates calls to different simulation backends.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import stim

import deltakit_explorer
from deltakit_explorer.enums._api_enums import DataFormat
from deltakit_explorer.types._types import LeakageFlags, Measurements


def simulate_with_stim(
    stim_circuit: str | stim.Circuit,
    shots: int,
    result_file: str | Path | None = None,
    client: deltakit_explorer.Client | None = None,
) -> tuple[Measurements, LeakageFlags | None]:
    """This method simulates the circuit using
    a Clifford STIM simulator. Clifford simulation is fast,
    and supports large numbers of qubits, but is limited in
    the set of gates one may use, and noise mechanisms.

    Args:
        stim_circuit (str | stim.Circuit):
            Circuit in STIM language.
        shots (int):
            Number of shots.
        result_file (Optional[str | Path]):
            If provided, saves the data to this file. If None,
            saves them in RAM. This is useful, when you run
            large local simulations, and want to save measurements
            directly to the disk. Not supported for cloud calls.
        client (Optional[Client]):
            If provided, performs the simulation using the `client`.
            Required if `stim_circuit` includes leakage.
            Currently implemented only when `result_file` is ``None``.

    Returns:
        Tuple[Measurements, Optional[LeakageFlags]]:
            (Measurements, Leakage). Leakage may be None.


    Examples:
        Running a circuit 1000 times and saving to a `b8` file::

            meas = simulation.simulate_with_stim(
                stim_circuit=noisy_circuit,
                shots=1000,
                result_file="results.b8",
            )

        Running a circuit 1000 times and saving to RAM::

            meas = simulation.simulate_with_stim(
                stim_circuit=noisy_circuit,
                shots=1000,
            )

    """

    if shots < 0:
        msg = "Please specify a non-negative number of shots."
        raise ValueError(msg)

    if client is not None:
        circuit = str(stim_circuit.as_stim_circuit()) if not isinstance(stim_circuit, str) else stim_circuit
        if result_file is not None:
            msg = "Use of `client` is currently incompatible with `result_file`."
            raise NotImplementedError(msg)
        measurements, leakage_flags = client.simulate_stim_circuit(circuit, shots=shots)
        return measurements, leakage_flags

    circuit = stim.Circuit(stim_circuit) if isinstance(stim_circuit, str) else stim_circuit
    sampler = circuit.compile_sampler()
    # E.g. if we under 100MB, save to RAM
    # result_size = circuit.num_measurements * shots
    if result_file is None:
        result = sampler.sample(shots=shots).astype(np.uint8)
        return Measurements(result, data_format=DataFormat.B8), None
    data_format = DataFormat.B8
    sampler.sample_write(
        shots=shots,
        filepath=str(result_file),
        format=data_format.value,
    )
    return Measurements(
        Path(result_file),
        data_format=data_format,
        data_width=circuit.num_measurements
    ), None
