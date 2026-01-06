# (c) Copyright Riverlane 2020-2025.
"""
This module provides a function which constructs a `deltakit.circuit.Circuit`
from a sequence of stages.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from deltakit_circuit import Circuit

from deltakit_explorer.codes._css._css_stage import CSSStage
from deltakit_explorer.codes._css._detectors import get_stage_transition_circuit
from deltakit_explorer.qpu._circuits import merge_layers


def experiment_circuit(experiment: Sequence[CSSStage]) -> Circuit:
    """
    Return a noiseless Circuit for an experiment represented by the input
    Sequence of CSSStage instances.

    Parameters
    ----------
    experiment : Sequence[CSSStage]
        Sequence of CSSStage instances that defines the experiment.

    Returns
    -------
    Circuit
        Noiseless experiment circuit.

    Raises
    ------
    ValueError
        If experiment contains fewer than 3 CSSStage instances.
    ValueError
        If experiment doesn't start with a CSSStage which consists only of qubit
        resets.
    ValueError
        If experiment doesn't end with a CSSStage which consists only of qubit
        measurements and observable additions.
    """
    if len(experiment) < 3:
        msg = "Experiment should contain at least three stages."
        raise ValueError(msg)
    if not experiment[0].resets_only:
        raise ValueError(
            "Experiment should start with a CSSStage which"
            + " consists only of qubit resets."
        )
    if not experiment[-1].allowable_final_stage:
        msg = (
            "Experiment should end with a CSSStage with properties as described in "
            "allowable_final_stage."
        )
        raise ValueError(msg)
    circuit = experiment[0].remaining_rounds

    # Get the detectors for stage transitions.
    stage_transition_detectors = [
        get_stage_transition_circuit(previous_stage, current_stage)
        for previous_stage, current_stage in itertools.pairwise(experiment)
    ]
    for stage, stage_transition_detector in zip(
        experiment[1:], stage_transition_detectors
    ):
        # Apply first round with the observables
        circuit.append_layers(stage.first_round)
        # Apply transition detectors
        circuit.append_layers(stage_transition_detector)
        # Apply remaining rounds
        circuit.append_layers(stage.remaining_rounds)

    # Compress circuit
    return merge_layers(circuit, break_repeat_blocks=True)
