# (c) Copyright Riverlane 2020-2025.
"""Module which provides ways to identify instructions to deltakit_stim."""

from typing import NamedTuple

import deltakit_stim


class NoiseDeltakitStimIdentifier(NamedTuple):
    """Collection of information which uniquely identifies a noise channel to
    deltakit_stim.

    Differs from the `GateDeltakitStimIdentifier` in that noise channels can
    have multiple probabilities which need to be taken into account."""

    deltakit_stim_string: str
    probabilities: tuple[float, ...]
    tag: str | None = None


class AppendArguments(NamedTuple):
    """Collection of items used when appending to a deltakit_stim circuit. This object
    could be destructured and passed to the `deltakit_stim.Circuit.append` method.
    """

    deltakit_stim_string: str
    deltakit_stim_targets: tuple[deltakit_stim.GateTarget, ...]
    arguments: tuple[float, ...]
    tag: str | None = None
