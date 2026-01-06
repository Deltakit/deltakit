# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain
from typing import Any, TypeAlias

import deltakit_circuit as sp
import numpy as np
import numpy.typing as npt
import stim
from deltakit_core.decoding_graphs import OrderedSyndrome

from deltakit_decode.noise_sources._generic_noise_sources import (
    BatchErrorGenerator,
    MonteCarloNoise,
)

StimErrorT: TypeAlias = tuple[OrderedSyndrome, tuple[bool, ...]]


def give_empty(_):
    """Function which just returns an empty list."""
    return []


class StimNoise(MonteCarloNoise[stim.Circuit,
                                StimErrorT], ):
    """A noise model that takes a set of noise profiles
    that can be applied to a lestim circuit. These noise profiles
    should be defined as callables that will be processed by deltakit_circuit.

    Parameters
    ----------
    before_gate_noise_profile : Optional[sp.NoiseProfile | Iterable[sp.NoiseProfile]],
    optional
        Profile of noise to run before gates, by default None.
    after_gate_noise_profile : Optional[sp.NoiseProfile | Iterable[sp.NoiseProfile]],
    optional
        Profile of noise to run after gates, by default None.
    gate_replacement_policy : Optional[sp.GateReplacementPolicy], optional
        Rules to apply noise by replacing specific gates, by default None.
    batch_size : int, optional
        Number of syndromes to generate in a parallel batch, by default 1024.
    """

    def __init__(self,
                 before_gate_noise_profile:
                     sp.NoiseProfile | Iterable[sp.NoiseProfile] | None = None,
                 after_gate_noise_profile:
                     sp.NoiseProfile | Iterable[sp.NoiseProfile] | None = None,
                 gate_replacement_policy:
                     sp.GateReplacementPolicy | None = None,
                 batch_size=1024):
        self._batch_size = batch_size
        self._before_gate_noise_profile: \
            sp.NoiseProfile | Iterable[sp.NoiseProfile] = \
            before_gate_noise_profile \
            if before_gate_noise_profile is not None else give_empty
        self._after_gate_noise_profile: \
            sp.NoiseProfile | Iterable[sp.NoiseProfile] = \
            after_gate_noise_profile \
            if after_gate_noise_profile is not None else give_empty
        self._gate_replacement_policy: \
            sp.GateReplacementPolicy = \
            gate_replacement_policy \
            if gate_replacement_policy is not None else {}
        self._batch_size = batch_size

    def permute_stim_circuit(self, stim_circuit: stim.Circuit) -> stim.Circuit:
        """Apply noise to a lestim circuit

        Parameters
        ----------
        stim_circuit : stim.Circuit
            The lestim circuit to manipulate with StimNoise's noise profiles

        Returns
        -------
        stim.Circuit
            A new lestim circuit with noise applied via the noise profiles
        """
        circuit = sp.Circuit.from_stim_circuit(stim_circuit)
        circuit.remove_noise()
        circuit.apply_gate_noise(
            self._before_gate_noise_profile, sp.Circuit.LayerAdjacency.BEFORE)
        circuit.apply_gate_noise(
            self._after_gate_noise_profile, sp.Circuit.LayerAdjacency.AFTER)
        circuit.replace_gates(self._gate_replacement_policy)
        return circuit.as_stim_circuit()

    def error_generator(
        self,
        code_data: stim.Circuit,
        seed: int | None = None
    ) -> Iterator[StimErrorT]:
        stim_circuit = self.permute_stim_circuit(code_data)
        sampler = stim_circuit.compile_detector_sampler(seed=seed)
        num_observables = stim_circuit.num_observables
        while True:
            stim_batch: npt.NDArray[np.uint8] = sampler.sample(self._batch_size,
                                                               append_observables=True)
            yield from stim_batch_to_decode_batch(stim_batch, num_observables)

    def build_batch_error_generator(
        self,
        code_data: stim.Circuit,
        seed: int | None = None
    ) -> BatchErrorGenerator:
        """Given some representation of a code, return a generator of batches of errors
        for that code.
        """
        stim_circuit = self.permute_stim_circuit(code_data)
        sampler = stim_circuit.compile_detector_sampler(seed=seed)
        return BatchErrorGenerator(
            lambda num_shots: sampler.sample(num_shots, separate_observables=True))

    def __add__(self, other):
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "StimNoise"


class OptionedStim(StimNoise):
    """A class with the ability to manipulate lestim circuits
    with after clifford gate depolarisation, before measure
    flip probability and after reset flip probability. For
    more information on these noise profiles see:
    https://github.com/quantumlib/Stim/blob/main/doc/
    python_api_reference_vDev.md#stim.Circuit.generated

    Parameters
    ----------
    after_clifford_depolarisation : float, optional
        Rate at which to depolarize after Clifford gates, by default 0.0.
    before_round_data_depolarisation : float, optional
        Rate at which to depolarize before a QEC round, by default 0.0.
    before_measure_flip_probability : float, optional
        Rate at which to flip measurement results, by default 0.0.
    after_reset_flip_probability : float, optional
        Rate at which to flip reset qubits, by default 0.0.

    """

    def __init__(self, after_clifford_depolarisation: float = 0.0,
                 before_round_data_depolarisation: float = 0.0,
                 before_measure_flip_probability: float = 0.0,
                 after_reset_flip_probability: float = 0.0):
        self._after_clifford_depolarisation = after_clifford_depolarisation
        self._before_measure_flip_probability = before_measure_flip_probability
        self._after_reset_flip_probability = after_reset_flip_probability

        after_gate_noise = []
        before_gate_noise = []

        if after_clifford_depolarisation > 0.0:
            after_gate_noise += sp.after_clifford_depolarisation(
                after_clifford_depolarisation)

        if before_round_data_depolarisation > 0.0:
            raise NotImplementedError()

        if before_measure_flip_probability > 0.0:
            before_gate_noise += sp.before_measure_flip_probability(
                before_measure_flip_probability)

        if after_reset_flip_probability > 0.0:
            after_gate_noise += sp.after_reset_flip_probability(
                after_reset_flip_probability)

        super().__init__(after_gate_noise_profile=after_gate_noise,
                         before_gate_noise_profile=before_gate_noise)

    def __repr__(self) -> str:
        return "OptionedStim"

    def field_values(self) -> dict[str, Any]:
        base_dict = super().field_values()
        base_dict["after_clifford_depolarisation"] = \
            self._after_clifford_depolarisation
        base_dict["before_measure_flip_probability"] = \
            self._before_measure_flip_probability
        base_dict["after_reset_flip_probability"] = \
            self._after_reset_flip_probability
        return base_dict


class ToyNoise(StimNoise):
    """A noise model which adds:

    * Depolarise 1 channel after every one qubit gate, reset gate and
      measurement gates (with the exception of the MPP gate)
    * Inverse noise after every two qubit gate and MPP gate. Inverse
      noise consists of a depolarise 2 channel after every two qubit gate
      in a gate layer and a depolarise 1 channel after all qubits that
      are not acted on in the gate layer
    * Depolarise1 channels have an error probability of `p_physical/10`
      while depolarise2 channels have an error probability of `p_physical`

    Parameters
    ----------
    p_physical : float
        Single parameter used to define the rate of occurrence of several
        noise channels.
    """

    def __init__(self, p_physical: float):
        self._p_physical = p_physical

        depolarise1_generator: sp.NoiseChannelGen = \
            sp.noise_channels.Depolarise1.generator_from_prob(p_physical/10)
        depolarise2_generator: sp.NoiseChannelGen = \
            sp.noise_channels.Depolarise2.generator_from_prob(p_physical)

        after_gate_noise = [
            lambda noise_context:
                depolarise1_generator(
                    list(chain.from_iterable(
                        noise_context.gate_layer_qubits(target_gate)
                        for target_gate in (sp.gates.OneQubitCliffordGate,
                                            sp.gates.OneQubitResetGate,
                                            sp.gates.OneQubitMeasurementGate)
                    ))),
        ]

        after_gate_noise += [
            sp.noise_profile_with_inverted_noise(
                target_gate_t,
                target_noise_generator=depolarise2_generator)
            for target_gate_t in [*sp.gates.TWO_QUBIT_GATES, sp.gates.MPP]
        ]

        after_gate_noise.append(
            sp.noise_profile_with_inverted_noise(
                None,
                inverse_noise_generator=depolarise1_generator)
        )

        super().__init__(
            after_gate_noise_profile=after_gate_noise,
            gate_replacement_policy=sp.measurement_noise_profile(p_physical))

    def __repr__(self) -> str:
        return "ToyNoise"

    def field_values(self) -> dict[str, Any]:
        base_dict = super().field_values()
        base_dict["p_physical"] = self._p_physical
        return base_dict


class SampleStimNoise(StimNoise):
    """A noise model that uses a stim file, without modification, to generate noise.

    By default, uses batch model to generate noise to make use of stim parallelism.
    This batching is hidden from the user, and this model provides the same interface
    as other sources.
    """

    def __init__(self, batch_size=1024):
        super().__init__(batch_size=batch_size)

    def permute_stim_circuit(self, stim_circuit: stim.Circuit) -> stim.Circuit:
        return stim_circuit

    def __repr__(self) -> str:
        return "SampleStimNoise"


def stim_batch_to_decode_batch(
    detector_batch: npt.NDArray[np.uint8],
    num_observables: int = 0,
) -> Iterator[tuple[OrderedSyndrome, tuple[bool, ...]]]:
    """Convert a Stim detector batch, optionally including observables, to a batch
    appropriate for use with deltakit-decode, with deltakit-decode syndrome objects and the observables split
    out as a tuple of booleans.

    Parameters
    ----------
    detector_batch : npt.NDArray[np.uint8]
        2D array, outer structure is list of samples, inner structure is a bitstring
        of detectors, with `num_observables` observables appended.
    num_observables : int, optional
        Number of observables in the bitstring of each sample, by default 0.

    Yields
    ------
    Tuple[OrderedSyndrome, Tuple[bool, ...]]
        The syndrome and reference logicals for the given sample.
    """
    for sample in detector_batch:
        if num_observables:
            detectors, observables = sample[:-num_observables], sample[-num_observables:]
        else:
            detectors, observables = sample, []
        yield (OrderedSyndrome.from_bitstring(detectors),
               tuple(bool(observable) for observable in observables))
