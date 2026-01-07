# (c) Copyright Riverlane 2020-2025.
"""Module which provides a layer in a circuit which contains only noise."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from itertools import chain, permutations
from typing import Generic

import stim

from deltakit_circuit._qubit_identifiers import Qubit, T, U
from deltakit_circuit._qubit_mapping import default_qubit_mapping
from deltakit_circuit._stim_identifiers import AppendArguments, NoiseStimIdentifier
from deltakit_circuit._stim_version_compatibility import is_stim_tag_feature_available
from deltakit_circuit.noise_channels import (
    Leakage,
    Relax,
    _LeakageNoise,
    _NoiseChannel,
    _UncorrelatedNoise,
)
from deltakit_circuit.noise_channels._correlated_noise import (
    CorrelatedError,
    ElseCorrelatedError,
    _CorrelatedNoise,
)
from deltakit_circuit.noise_channels._depolarising_noise import Depolarise1, Depolarise2
from deltakit_circuit.noise_channels._pauli_noise import (
    PauliChannel1,
    PauliChannel2,
    PauliXError,
    PauliYError,
    PauliZError,
)

UNCORRELATED_NOISE_CHANNELS = (
    PauliXError,
    PauliYError,
    PauliZError,
    PauliChannel1,
    PauliChannel2,
    Depolarise1,
    Depolarise2,
    Leakage,
    Relax,
)
CORRELATED_NOISE_CHANNELS = (CorrelatedError, ElseCorrelatedError)
ALL_NOISE_CHANNELS = UNCORRELATED_NOISE_CHANNELS + CORRELATED_NOISE_CHANNELS


class NoiseLayer(Generic[T]):
    """Class which represents noise in a circuit."""

    def __init__(
        self, noise_channels: _NoiseChannel | Iterable[_NoiseChannel] | None = None
    ):
        self._uncorrelated_noise_channels: list[_UncorrelatedNoise | _LeakageNoise] = []
        self._correlated_noise_channels: list[_CorrelatedNoise] = []
        if noise_channels is not None:
            self.add_noise_channels(noise_channels)

    @property
    def noise_channels(self) -> tuple[_NoiseChannel, ...]:
        """Get all noise channels in this layer."""
        return tuple(
            chain(self._uncorrelated_noise_channels, self._correlated_noise_channels)
        )

    @property
    def qubits(self) -> frozenset[Qubit[T]]:
        """Get all the qubits in this noise layer."""
        return frozenset(
            chain.from_iterable(
                noise_channel.qubits
                for noise_channel in chain(
                    self._uncorrelated_noise_channels, self._correlated_noise_channels
                )
            )
        )

    def transform_qubits(self, id_mapping: Mapping[T, U]):
        """
        Transform all noise channels in this layer according to the id mapping.
        No transformation is performed if the qubits id is not in the mapping.

        Parameters
        ----------
        id_mapping : Mapping[T, U]
            A mapping of qubit types to other qubit types
        """
        for noise_channel in chain(
            self._uncorrelated_noise_channels, self._correlated_noise_channels
        ):
            noise_channel.transform_qubits(id_mapping)

    def add_noise_channels(
        self, noise_channels: _NoiseChannel | Iterable[_NoiseChannel]
    ):
        """Add noise channels to this noise layer.

        Parameters
        ----------
        noise_channel : _NoiseChannel | Iterable[_NoiseChannel]
            The noise channel to add.
        """
        noise_channels = (
            (noise_channels,)
            if isinstance(noise_channels, ALL_NOISE_CHANNELS)
            else noise_channels
        )
        for noise_channel in noise_channels:
            if isinstance(noise_channel, UNCORRELATED_NOISE_CHANNELS):
                self._uncorrelated_noise_channels.append(noise_channel)
            else:
                self._correlated_noise_channels.append(noise_channel)

    def _collect_noise_channels(
        self, qubit_mapping: Mapping[Qubit[T], int]
    ) -> list[AppendArguments]:
        grouped_noise: defaultdict[NoiseStimIdentifier, list[stim.GateTarget]] = (
            defaultdict(list)
        )
        for noise_channel in self._uncorrelated_noise_channels:
            grouped_noise[noise_channel.stim_identifier].extend(
                noise_channel.stim_targets(qubit_mapping)
            )
        uncorrelated_noise = [
            AppendArguments(stim_string, tuple(targets), probabilities, tag)
            for (stim_string, probabilities, tag), targets in grouped_noise.items()
        ]
        correlated_noise = [
            AppendArguments(
                noise_channel.stim_string,
                noise_channel.stim_targets(qubit_mapping),
                noise_channel.probabilities,
                noise_channel.tag,
            )
            for noise_channel in self._correlated_noise_channels
        ]
        return uncorrelated_noise + correlated_noise

    def permute_stim_circuit(
        self,
        stim_circuit: stim.Circuit,
        qubit_mapping: Mapping[Qubit[T], int] | None = None,
    ):
        """Updates stim_circuit with the stim circuit which contains the noise
        channels specified in this NoiseLayer.

        Parameters
        ----------
        stim_circuit : stim.Circuit
            The stim circuit to be updated with the stim representation of
            this noise layer.

        qubit_mapping : Mapping[Qubit[T], int] | None, optional
            A mapping between each qubit in this layer and an integer which is
            necessary for outputting a stim circuit. By default None which
            means a default mapping is used.
        """
        qubit_mapping = (
            default_qubit_mapping(self.qubits)
            if qubit_mapping is None
            else qubit_mapping
        )
        for stim_string, targets, probabilities, tag in self._collect_noise_channels(
            qubit_mapping
        ):
            kwargs = (
                {"tag": tag}
                if tag is not None and is_stim_tag_feature_available()
                else {}
            )
            stim_circuit.append(stim_string, targets, probabilities, **kwargs)

    def approx_equals(  # noqa: PLR0911
        self,
        other: object,
        *,
        rel_tol: float = 1e-9,
        abs_tol: float = 0,
    ) -> bool:
        """Determine whether two noise layers are approximately equal
        within a tolerance. The tolerance accounts for small differences
        in the error probabilities of noise channels. All other properties
        must be equal.

        Parameters
        ----------
        other : object
            The other object to which to compare this gate layer.
        rel_tol : float
            The allowed relative difference between the error probabilities
            of two noise channels, if this is larger than that calculated from
            abs_tol. Has the same meaning as in math.isclose.
            By default, 1e-9.
        abs_tol : float, optional
            The allowed absolute difference between the error probabilities
            of two noise channels, if this is larger than that calculated
            from rel_tol. Has the same meaning as in math.isclose.
            By default, 0.0.

        Returns
        -------
        bool
            Whether the two noise layers are approximately equal.
        """
        # pylint: disable=protected-access

        if not isinstance(other, NoiseLayer):
            return False

        if len(self._correlated_noise_channels) != len(
            other._correlated_noise_channels
        ):
            return False

        if not all(
            self_channel.approx_equals(other_channel, rel_tol=rel_tol, abs_tol=abs_tol)
            for self_channel, other_channel in zip(
                self._correlated_noise_channels,
                other._correlated_noise_channels,
                strict=True,
            )
        ):
            return False

        self_channel_dict = defaultdict(list)
        for channel in self._uncorrelated_noise_channels:
            self_channel_dict[channel.qubits].append(channel)

        other_channel_dict = defaultdict(list)
        for channel in other._uncorrelated_noise_channels:
            other_channel_dict[channel.qubits].append(channel)

        if set(self_channel_dict.keys()) != set(other_channel_dict.keys()):
            return False

        for qubits, self_channels in self_channel_dict.items():
            other_channels = other_channel_dict[qubits]
            # Early exit if we can
            if len(self_channels) != len(other_channels):
                return False
            for other_channels_order in permutations(
                other_channels, len(other_channels)
            ):
                if all(
                    self_channel.approx_equals(
                        other_channel, rel_tol=rel_tol, abs_tol=abs_tol
                    )
                    for self_channel, other_channel in zip(
                        self_channels, other_channels_order, strict=True
                    )
                ):
                    break
            else:
                return False

        return True

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, NoiseLayer)
            and Counter(self._uncorrelated_noise_channels)
            == Counter(other._uncorrelated_noise_channels)
            and self._correlated_noise_channels == other._correlated_noise_channels
        )

    def __hash__(self) -> int:
        msg = (
            "Hash is expected to be implemented in constant time but there is not easy "
            "way of achieving that complexity with the current NoiseLayer internals. "
            "If you get this error, please open an issue on "
            "https://github.com/Deltakit/deltakit/issues/new/choose."
        )
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        indent = 4 * " "
        noise_layer_lines = ["NoiseLayer(["]
        noise_layer_lines.extend(
            f"{indent}{noise_channel!r}" for noise_channel in self.noise_channels
        )
        noise_layer_lines.append("])")
        return "\n".join(noise_layer_lines)
