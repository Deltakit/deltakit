# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from itertools import islice
from pathlib import Path
from typing import Any, TypeAlias
from warnings import warn

import numpy as np
import numpy.typing as npt
import stim
from deltakit_core.data_formats import b8_to_logical_flip, b8_to_syndromes
from deltakit_core.decoding_graphs import (
    DecodingHyperEdge,
    EdgeT,
    HyperMultiGraph,
    OrderedSyndrome,
)

from deltakit_decode._abstract_matching_decoders import GraphDecoder
from deltakit_decode._base_reporter import BaseReporter
from deltakit_decode.analysis._decoder_manager import (
    DecoderManager,
    NoiseModelDecoderManager,
)
from deltakit_decode.noise_sources import SampleStimNoise
from deltakit_decode.noise_sources._generic_noise_sources import NoiseModel

StimOutput: TypeAlias = tuple[OrderedSyndrome, tuple[bool, ...]]
StimBatchOutput: TypeAlias = tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]


class StimDecoderManager(
        NoiseModelDecoderManager[StimOutput, stim.Circuit, tuple[bool, ...],
                                 StimBatchOutput, npt.NDArray[np.uint8]]):
    """Decoder manager to support Stim circuits being used for noise generation, and
    Stim observables to define decoder success.

    Parameters
    ----------
    stim_noise_circuit : stim.Circuit
        Stim circuit to use to inform how noise samples are generated.
    decoder : GraphDecoder
        Decoder to use to decode generated shots.
    noise_model : NoiseModel[stim.Circuit, StimOutput] | None, optional
        Stim circuit based noise model to use, when `None` an instance of
        `SampleStimNoise` which will just directly take samples from the
        `stim_noise_circuit` is used.
    reporters : Optional[List[BaseReporter]], optional
        Optional list of reporters to give extra information about a decoder,
        by default None.
    metadata : Optional[Dict[str, str]], optional
        Metadata to associate with this experiment, by default None.
    seed : Optional[int], optional
        Optional seed for use with given noise sources, by default None which results in
        random seed generation.
    """

    def __init__(
            self,
            stim_noise_circuit: stim.Circuit,
            decoder: GraphDecoder,
            noise_model: NoiseModel[stim.Circuit, StimOutput] | None = None,
            reporters: list[BaseReporter] | None = None,
            metadata: dict[str, str] | None = None,
            seed: int | None = None,
            batch_size: int = int(1e4)):
        if noise_model is None:
            noise_model = SampleStimNoise()
        super().__init__(noise_model,
                         len(decoder.logicals),
                         reporters,
                         metadata,
                         seed=seed,
                         batch_size=batch_size)
        self._stim_noise_circuit = stim_noise_circuit
        self._decoder = decoder

    @property
    def errors_per_logical(self) -> list[int]:
        """Uses self._empirical_decoding_error_distribution to get the number of
        fails for each logical in the order they appear in
        self._decoder.logicals."""
        return self._empirical_decoding_error_distribution.fails_per_logical.tolist()

    def _analyse_correction(self, error: StimOutput, correction: tuple[bool, ...]
                            ) -> bool:
        _, target_logical_flip = error
        return target_logical_flip != correction

    def _decode_from_error(self, error: StimOutput) -> tuple[bool, ...]:
        syndrome, target_logical_flip = error
        correction = self._decoder.decode_to_logical_flip(syndrome)
        self._empirical_decoding_error_distribution.record_error(correction,
                                                                 target_logical_flip)
        return correction

    def _decode_batch_from_error(self, errors: StimBatchOutput) -> npt.NDArray[np.uint8]:
        syndrome_batch, actual_observables = errors
        predicted_observables = self._decoder.decode_batch_to_logical_flip(syndrome_batch)
        self._empirical_decoding_error_distribution.batch_record_errors(
            predicted_observables, actual_observables)
        return predicted_observables

    def get_reporter_results(self) -> dict[str, Any]:
        analysis_results: dict[str, Any] = {"decoder": str(self._decoder)}
        analysis_results.update(self._noise_model.field_values())
        analysis_results.update(super().get_reporter_results())
        if len(self._decoder.logicals) > 1:
            errors_per_logical = {
                f"fails_log_{i}": fails
                for i, fails in enumerate(self.errors_per_logical)
            }
            analysis_results.update(errors_per_logical)
        return analysis_results

    def _get_code_data(self) -> stim.Circuit:
        return self._stim_noise_circuit

    def __str__(self) -> str:
        return f"{self._decoder}_{self._noise_model}"

    @property
    def error_distribution_over_logicals(self) -> dict[tuple[bool, ...], int]:
        """Getter that provides insight into the distribution of failures over
        multiple logicals.

        Examples
        --------
            {(False, False, False): 4, (False, False, True): 1, ...,
            (True, True, True): 2}

            indicates that there were 4 cases where the
            correction and error did not differ at all, 1 case where the third
            logical alone differed and 2 cases where all the logicals differed.

        Returns
        -------
        Dict[Tuple[bool, ...], int]
            A dictionary that describes the distribution of failures across
            all combinations of logicals.
        """
        return self._empirical_decoding_error_distribution.to_dict()


class B8DecoderManager(DecoderManager):
    """Decoder manager to run experiments defined by a b8 input for syndromes
    and target logical flips.

    The b8 data formats were proposed by Google Quantum AI and can be read
    about here: https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#b8

    Parameters
    ----------
    syndrome_b8_file : Path | bytes
        Path to the file containing input syndromes in b8 format or a
        bytes object that stores b8 data
    logical_flip_b8_file : Path | bytes
        Path to the file containing input logical flips in b8 format or a
        bytes object that stores b8 data
    decoder : GraphDecoder
        Decoder to use for decoding.
    reporters : Optional[List[BaseReporter]], optional
        Reporters are not supported by this decoder manager, by default None.
    metadata : Optional[Dict[str, Any]], optional
        Metadata to associate with this experiment, by default None.
    """

    def __init__(
            self,
            syndrome_b8_input: Path | bytes,
            logical_flip_b8_input: Path | bytes,
            decoder: GraphDecoder,
            reporters: list[BaseReporter] | None = None,
            metadata: dict[str, Any] | None = None):
        super().__init__(len(decoder.logicals), reporters, metadata)
        if reporters:
            warn("Reporters on a B8DecoderManager will not be used.", stacklevel=2)
        self._syndrome_b8_input = syndrome_b8_input
        self._logical_flip_b8_input = logical_flip_b8_input
        self._decoder = decoder

    def run_single_shot(self) -> bool:
        raise NotImplementedError()

    def run_batch_shots(self, batch_limit: int | None) -> tuple[int, int]:
        detector_num = len(self._decoder.decoding_graph.nodes) - \
            len(self._decoder.decoding_graph.boundaries)
        num_logicals = len(self._decoder.logicals)
        syndrome_generator = b8_to_syndromes(self._syndrome_b8_input,
                                             detector_num)
        target_generator = b8_to_logical_flip(self._logical_flip_b8_input, num_logicals)
        syndrome_and_targets: Iterable[tuple[OrderedSyndrome, tuple[bool, ...]]]
        syndrome_and_targets = zip(syndrome_generator, target_generator)

        if batch_limit is not None:
            syndrome_and_targets = islice(syndrome_and_targets, batch_limit)

        for syndrome, target in syndrome_and_targets:
            correction = self._decoder.decode_to_logical_flip(syndrome)
            self._empirical_decoding_error_distribution.record_error(correction, target)

        return self.shots, self.fails


class GraphDecoderManager(NoiseModelDecoderManager[
    AbstractSet[EdgeT], HyperMultiGraph, tuple[bool, ...],
        list[AbstractSet[EdgeT]], list[tuple[bool, ...]]]):
    """Decoder manager for a graph decoder with an edge-based noise model. In
    this representation, an edge corresponds to a possible error event and
    the nodes of the edge correspond to the syndromes triggered by the error.
    """

    def __init__(
            self,
            noise_model: NoiseModel,
            decoder: GraphDecoder[HyperMultiGraph],
            decoding_graph: HyperMultiGraph | None = None,
            logicals: Sequence[AbstractSet[DecodingHyperEdge | int]] | None = None,
            reporters: list[BaseReporter] | None = None,
            metadata: dict[str, str] | None = None,
            seed: int | None = None,
            batch_size: int = int(1e4)):
        super().__init__(noise_model,
                         len(logicals
                             if logicals is not None
                             else decoder.logicals),
                         reporters,
                         metadata,
                         seed=seed,
                         batch_size=batch_size)
        self._decoder = decoder
        self._decoding_graph = decoding_graph or decoder.decoding_graph
        self._logicals = logicals if logicals is not None else decoder.logicals

    @property
    def errors_per_logical(self) -> list[int]:
        """Uses self._empirical_decoding_error_distribution_over_logicals to get
        the number of fails for each logical in the order they appear in
        self._decoder.logicals."""
        return self._empirical_decoding_error_distribution.fails_per_logical.tolist()

    def _analyse_correction(self,
                            error: AbstractSet[EdgeT],
                            correction: tuple[bool, ...],
                            ) -> bool:
        target_logical_flip = tuple(len(error & logical) % 2 == 1
                                for logical in self._logicals)
        return target_logical_flip != correction

    @property
    def error_distribution_over_logicals(self) -> dict[tuple[bool, ...], int]:
        """Getter that provides insight into the distribution of failures over
        multiple logicals.

        Examples
        --------
            {(False, False, False): 4, (False, False, True): 1, ...,
            (True, True, True): 2}

            indicates that there were 4 cases where the correction and error did
            not differ at all, 1 case where the third logical alone differed and
            2 cases where all the logicals differed.

        Returns
        -------
        Dict[Tuple[bool, ...], int]
            A dictionary that describes the distribution of failures across
            all combinations of logicals."""
        return self._empirical_decoding_error_distribution.to_dict()

    def _decode_from_error(self, error: AbstractSet[EdgeT]) -> tuple[bool, ...]:
        syndrome = self._decoding_graph.error_to_syndrome(error)
        correction = self._decoder.decode_to_logical_flip(syndrome)
        target_logical_flip = tuple(len(error & logical) % 2 == 1
                                    for logical in self._logicals)
        self._empirical_decoding_error_distribution.record_error(correction,
                                                                 target_logical_flip)
        return correction

    def _decode_batch_from_error(self, errors: list[AbstractSet[EdgeT]]
                                 ) -> list[tuple[bool, ...]]:
        return [self._decode_from_error(error) for error in errors]

    def get_reporter_results(self) -> dict[str, Any]:
        analysis_results = {"decoder": str(self._decoder)}
        analysis_results.update(self._noise_model.field_values())
        if len(self._decoder.logicals) > 1:
            errors_per_logical: dict[str, Any] = {
                f"fails_log_{i}": fails
                for i, fails in enumerate(self.errors_per_logical)
            }
            analysis_results.update(errors_per_logical)
        analysis_results.update(super().get_reporter_results())
        return analysis_results

    def _get_code_data(self) -> HyperMultiGraph:
        return self._decoder.decoding_graph

    def __str__(self) -> str:
        return f"{self._decoder}_{self._noise_model}"
