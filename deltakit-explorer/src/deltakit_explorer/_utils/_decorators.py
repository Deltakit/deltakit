# (c) Copyright Riverlane 2020-2025.
"""This file contains decorators which guard client's input.
Using decorators is motivated by separation of logic with parameter
validation.
"""
from __future__ import annotations

import functools
import itertools
from typing import Any

import stim
import tqdm

from deltakit_explorer._utils._logging import Logging
from deltakit_explorer._utils._utils import HTTP_PACKET_LIMIT
from deltakit_explorer.data._data_analysis import (
    get_decoding_request_size_estimate as get_dec_request_size,
)
from deltakit_explorer.data._data_analysis import (
    get_decoding_response_size_estimate as get_dec_size,
)
from deltakit_explorer.data._data_analysis import (
    get_simulation_response_size_estimate as get_sim_size,
)
from deltakit_explorer.data._data_analysis import has_leakage
from deltakit_explorer.enums._api_enums import DataFormat, DecoderType, QECECodeType
from deltakit_explorer.types._exceptions import ServerException
from deltakit_explorer.types._experiment_types import QECExperimentDefinition
from deltakit_explorer.types._types import (
    CircuitParameters,
    Decoder,
    DecodingResult,
    DetectionEvents,
    LeakageFlags,
    Measurements,
    ObservableFlips,
)


def validate_and_split_decoding(func):
    """Decorator which validates parameters for decoding,
    And warns about potential high consumption."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    @functools.wraps(func)
    def wrapper(
        obj,
        detectors: DetectionEvents,
        observables: ObservableFlips,
        decoder: Decoder,
        noisy_stim_circuit: str | stim.Circuit,
        leakage_flags: LeakageFlags | None = None,
    ):
        if decoder.decoder_type in {DecoderType.LCD} and leakage_flags is not None:
            Logging.warn(
                "Leakage-aware decoding has a heavy initialisation part. "
                "Big tasks may be cancelled by server timeout.",
                uid="decorator"
            )

        numpy_detectors = detectors.as_numpy()
        # numpy-based instance, to avoid multiple recreation
        detectors = DetectionEvents(
            numpy_detectors,
            DataFormat.B8,
            data_width=numpy_detectors.shape[1]
        )
        shots = numpy_detectors.shape[0]
        batch_size = min(100_000, shots)
        if decoder.decoder_type == DecoderType.LCD:  # LCD weakly avoids batching
            batch_size = shots
        while (
            get_dec_size(noisy_stim_circuit, batch_size, DataFormat.F01)
            > HTTP_PACKET_LIMIT
        ) and batch_size > 100:
            batch_size //= 2
        while (
            get_dec_request_size(noisy_stim_circuit, batch_size, DataFormat.B8)
            > HTTP_PACKET_LIMIT
        ) and batch_size > 100:
            batch_size //= 2
        batch_size = max(1, batch_size)
        if (
            decoder.decoder_type in {DecoderType.AC, DecoderType.BP_OSD}
            and len(str(noisy_stim_circuit)) > 25_000  # e.g. toric 5x5x5
            and numpy_detectors.shape[0] > 10_000  # shots
        ):
            Logging.warn(
                "If you decode with a hypergraph-decoder (AC, BP-OSD), "
                "decoding time may be significant and subject to the "
                "server timeout.",
                uid="decorator",
            )
        # batch decoding
        batch_detectors = detectors.to_batches(batch_size)
        batch_observables = observables.to_batches(batch_size)
        if leakage_flags is not None:
            batch_leakage = leakage_flags.to_batches(batch_size)
        else:
            batch_leakage = []
        results = []
        tasks: tqdm.tqdm[Any] | itertools.zip_longest[Any] = itertools.zip_longest(
            batch_detectors, batch_observables, batch_leakage
        )
        if batch_size < shots:
            iterations = (shots + batch_size - 1) // batch_size
            tasks = tqdm.tqdm(tasks, "decoding batches", total=iterations)

        for dets, obs, leakage in tasks:
            decoding_result = func(
                obj, dets, obs,
                decoder, noisy_stim_circuit, leakage,
            )
            results.append(decoding_result)
        return DecodingResult.combine(results)

    return wrapper


def validate_generation(func):
    """Decorator which validates circuit generation and
    warns if the task may consume a lot of server resources,
    which may lead to a timeout."""

    @functools.wraps(func)
    def wrapper(
        obj, experiment_definition: QECExperimentDefinition,
    ):
        parameters: CircuitParameters | None = experiment_definition.parameters
        if experiment_definition.code_type in {
            QECECodeType.ROTATED_PLANAR,
            QECECodeType.UNROTATED_PLANAR,
            QECECodeType.UNROTATED_TORIC
        } and parameters is not None and parameters.sizes is not None:
            area = functools.reduce(
                lambda a, b: a * b,
                parameters.sizes.sizes, 1)
            if area >= 21 ** 2 and experiment_definition.basis_gates is not None:
                Logging.warn(
                    "Circuit generation with a provided gate set may be slow for "
                    "big code patches. This may lead to a server timeout.",
                    uid="decorator",
                )
        if (
            experiment_definition.code_type in {QECECodeType.BIVARIATE_BICYCLE}
            and parameters is not None
            and parameters.matrix_specifications is not None
        ):
            param_l = parameters.matrix_specifications.param_l
            has_basis_gates = experiment_definition.basis_gates is not None
            if param_l >= 15 and has_basis_gates:
                Logging.warn(
                    "BBCode circuit generation with a provided gate set "
                    "may be slow. This may lead to a server timeout.",
                    uid="decorator",
                )
        return func(obj, experiment_definition)

    return wrapper


def _split_into_batches(total: int, batch_size: int) -> list[int]:
    shots_list = []
    while total > 0:
        shots_list.append(min(batch_size, total))
        total -= batch_size
    return shots_list


def validate_and_split_simulation(func):
    """Decorator which validates parameters for simulation,
    And warns about potential high resource consumption."""

    @functools.wraps(func)
    def wrapper(  # pragma: nocover
        obj,
        stim_circuit: str | stim.Circuit,
        shots: int,
    ):
        assert shots > 0, "Number of shots should be positive."
        batch_size = min(100_000, shots)  # initial guess
        while (
            get_sim_size(stim_circuit, batch_size, DataFormat.F01) > HTTP_PACKET_LIMIT
            and batch_size > 100
        ):
            batch_size //= 2
        batch_size = max(1, batch_size)

        # do not split if simulated locally
        if not has_leakage(str(stim_circuit)):
            batch_size = shots

        shots_list: list[int] | tqdm.tqdm[int] = _split_into_batches(shots, batch_size)

        if len(shots_list) > 1:
            Logging.warn(
                "Simulation will be performed in batches "
                f"of {batch_size} shots.",
                uid="decorator",
            )
        measurements = []
        leakages = []
        if len(shots_list) > 1:
            shots_list = tqdm.tqdm(shots_list, "simulation batches")
        for batch in shots_list:
            Logging.info(f"Simulating {batch} shots...", uid="decorator")
            meas, leak = func(obj, stim_circuit, batch)
            measurements.append(meas)
            leakages.append(leak)
        if (
            not all(leak is None for leak in leakages)
            and not all(leak is not None for leak in leakages)
        ):
            msg = (
                "Some of batches returned leakage information, "
                "and some did not. Data is inconsistent."
            )
            raise ServerException(
                msg
            )
        final_measurements = Measurements.combine(measurements)
        final_leakages = None
        if all(leak is not None for leak in leakages):
            final_leakages = LeakageFlags.combine(leakages)
        return (final_measurements, final_leakages)

    return wrapper
