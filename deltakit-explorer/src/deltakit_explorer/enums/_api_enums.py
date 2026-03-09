# (c) Copyright Riverlane 2020-2025.
"""Enums used across the client library."""

from __future__ import annotations

from enum import Enum


class APIEndpoints(Enum):
    """Names of API endpoints"""

    GENERATE_CIRCUIT = "generate_circuit"
    ADD_NOISE = "generate_noisy_stim_circuit"
    ADD_SI1000_NOISE = "generate_si1000_noisy_stim_circuit"
    DECODE = "decode"
    DECODE_LEAKAGE = "decode_leakage"
    DEFECT_RATES = "defect_rates"
    CORRELATION_MATRIX = "correlation_matrix"
    TRIM_CIRCUIT_AND_DETECTORS = "trim_circuit_and_detectors"
    SIMULATE_CIRCUIT = "simulation"


class DataFormat(Enum):
    """Represent common data representation formats.

    Supported values are ``F01``, ``B8``, ``CSV``, and ``TEXT``.
    """

    F01 = "01"
    B8 = "b8"
    CSV = "csv"
    TEXT = "text"


class DecoderType(Enum):
    """Decoder types, supported by a client. Supported values are:

    - ``MWPM``: Minimum Weight Perfect Matching [1]_
    - ``CC``: Collision Clustering [2]_
    - ``BELIEF_MATCHING``: Belief Matching [3]_
    - ``BP-OSD``: Belief Propagation - Ordered Statistics Decoding [4]_
    - ``AC``: Ambiguity Clustering [5]_
    - ``LCD``: Local Clustering Decoder [6]_

    References
    ----------
    .. [1] https://arxiv.org/abs/2303.15933
    .. [2] https://arxiv.org/abs/2309.05558
    .. [3] https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.031007
    .. [4] https://quantum-journal.org/papers/q-2021-11-22-585/
    .. [5] https://arxiv.org/abs/2406.14527
    .. [6] https://arxiv.org/abs/2411.10343

    """

    MWPM = "MWPM"
    CC = "CC"
    BELIEF_MATCHING = "BELIEF_MATCHING"
    BP_OSD = "BP_OSD"
    AC = "AC"
    LCD = "LCD"


class QECExperimentType(Enum):
    """Type of QEC experiment.

    Supported values are ``QUANTUM_MEMORY`` and ``STABILITY``.

    """

    QUANTUM_MEMORY = "QUANTUM_MEMORY"
    STABILITY = "STABILITY"


class QECECodeType(Enum):
    """Type of QEC code.

    Supported values are ``ROTATED_PLANAR``, ``UNROTATED_PLANAR``,
    ``REPETITION``, ``BIVARIATE_BICYCLE``, and ``UNROTATED_TORIC``.

    """

    ROTATED_PLANAR = "ROTATED_PLANAR"
    UNROTATED_PLANAR = "UNROTATED_PLANAR"
    REPETITION = "REPETITION"
    BIVARIATE_BICYCLE = "BIVARIATE_BICYCLE"
    UNROTATED_TORIC = "UNROTATED_TORIC"
