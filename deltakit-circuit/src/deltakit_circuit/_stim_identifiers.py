# (c) Copyright Riverlane 2020-2025.
"""Module which provides ways to identify instructions to stim."""

from typing import NamedTuple

import stim


class NoiseStimIdentifier(NamedTuple):
    """Collection of information which uniquely identifies a noise channel to
    stim. Differs from the `GateStimIdentifier` in that noise channels can
    have multiple probabilities which need to be taken into account."""

    stim_string: str
    probabilities: tuple[float, ...]
    tag: str | None = None


class AppendArguments(NamedTuple):
    """Collection of items used when appending to a stim circuit. This object
    could be destructured and passed to the `stim.Circuit.append` method.
    """

    stim_string: str
    stim_targets: tuple[stim.GateTarget, ...]
    arguments: tuple[float, ...]
    tag: str | None = None
