.. meta::
   :robots: noindex, nofollow

Reference API
=============

.. _api-deltakit-circuit:

``deltakit.circuit``
--------------------

Top Level
^^^^^^^^^

.. currentmodule:: deltakit.circuit

``deltakit.circuit`` provides classes to represent Stim circuit elements
and functions/methods for interacting with them. The following features
are imported directly from the top-level module.

Circuit Building Blocks
"""""""""""""""""""""""

.. autosummary::
    :toctree: _build/generated/

    Circuit
    GateLayer
    NoiseLayer
    Qubit
    Detector
    Observable
    ShiftCoordinates
    SweepBit
    MeasurementRecord

Paulis
""""""

.. autosummary::
    :toctree: _build/generated/

    PauliX
    PauliY
    PauliZ
    PauliProduct
    InvertiblePauliX
    InvertiblePauliY
    InvertiblePauliZ
    MeasurementPauliProduct

Typing
""""""

.. autosummary::
    :toctree: _build/generated/

    Layer
    GateReplacementPolicy
    NoiseChannelGen
    NoiseProfile

Circuit Manipulation and Noise Generation
"""""""""""""""""""""""""""""""""""""""""

.. autosummary::
    :toctree: _build/generated/

    trim_detectors
    after_clifford_depolarisation
    after_reset_flip_probability
    before_measure_flip_probability
    measurement_noise_profile
    noise_profile_with_inverted_noise

Other
"""""

.. autosummary::
    :toctree: _build/generated/

    Coordinate
    NoiseContext


.. _api-deltakit-circuit-gates:

``deltakit.circuit.gates``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: deltakit.circuit.gates

``deltakit.circuit.gates`` contains class-based representations of Stim gates.

One-Qubit Gates
"""""""""""""""

.. autosummary::
    :toctree: _build/generated/

    I
    H
    H_XY
    H_YZ
    X
    SQRT_X
    SQRT_X_DAG
    Y
    SQRT_Y
    SQRT_Y_DAG
    Z
    S
    S_DAG
    C_XYZ
    C_ZYX

Two-Qubit Gates
"""""""""""""""

.. autosummary::
    :toctree: _build/generated/

    CX
    CY
    CZ
    XCX
    XCY
    XCZ
    YCX
    YCY
    YCZ
    SWAP
    CXSWAP
    CZSWAP
    ISWAP
    ISWAP_DAG
    SQRT_XX
    SQRT_XX_DAG
    SQRT_YY
    SQRT_YY_DAG
    SQRT_ZZ
    SQRT_ZZ_DAG

Measurement Gates
"""""""""""""""""

.. autosummary::
    :toctree: _build/generated/

    MX
    MY
    MZ
    MRX
    MRY
    MRZ
    MPP

Reset Gates
"""""""""""

.. autosummary::
    :toctree: _build/generated/

    RX
    RY
    RZ

Leakage Gates
"""""""""""""

.. autosummary::
    :toctree: _build/generated/

    HERALD_LEAKAGE_EVENT

Abstract Gates
""""""""""""""

.. autosummary::
    :toctree: _build/generated/

    Gate
    OneQubitCliffordGate
    OneQubitGate
    TwoOperandGate
    OneQubitMeasurementGate
    OneQubitResetGate
    SymmetricTwoQubitGate
    PauliBasis

Gates Sets
""""""""""

.. autosummary::
    :toctree: _build/generated/

    ONE_QUBIT_GATES
    TWO_QUBIT_GATES
    ONE_QUBIT_MEASUREMENT_GATES
    MEASUREMENT_GATES
    RESET_GATES

.. _api-deltakit-circuit-noise_channels:

``deltakit.circuit.noise_channels``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: deltakit.circuit.noise_channels

``deltakit.circuit.noise_channels`` provides classes representing noise channels
and error models for quantum circuit composition.

.. autosummary::
    :toctree: _build/generated/

    MultiProbabilityNoiseChannel
    NoiseChannel
    OneProbabilityNoiseChannel
    OneQubitNoiseChannel
    TwoQubitNoiseChannel
    CorrelatedError
    ElseCorrelatedError
    Depolarise1
    Depolarise2
    Leakage
    Relax
    PauliChannel1
    PauliChannel2
    PauliXError
    PauliYError
    PauliZError

.. _api-deltakit-core:

``deltakit.core``
-----------------

.. currentmodule:: deltakit.core

``deltakit.core`` provides foundational data structures and primitives for
quantum error correction workflows, including bitstrings, syndromes, and decoding graphs.

.. autosummary::
    :toctree: _build/generated/

.. _api-deltakit-core-data_formats:

``deltakit.core.data_formats``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.core.data_formats`` provides utilities for converting and parsing
syndromes, measurements, and logical flips across standard formats (such as b8 and c64).

.. currentmodule:: deltakit.core.data_formats

.. autosummary::
    :toctree: _build/generated/

    b8_to_logical_flip
    b8_to_measurements
    b8_to_syndromes
    c64_to_addressed_input_words
    logical_flips_to_b8_file
    parse_01_to_logical_flips
    parse_01_to_syndromes
    split_input_data_to_c64
    syndromes_to_b8_file
    to_bytearray

.. _api-deltakit-core-decoding_graphs:

``deltakit.core.decoding_graphs``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.core.decoding_graphs`` provides graph representations, DEM parsers,
and topological utilities for constructing and analysing decoding graphs.

.. currentmodule:: deltakit.core.decoding_graphs

.. autosummary::
    :toctree: _build/generated/

    AnyEdgeT
    Bitstring
    change_graph_error_probabilities
    compute_graph_distance
    compute_graph_distance_for_logical
    DecodingCode
    DecodingEdge
    DecodingHyperEdge
    DecodingHyperGraph
    DecodingHyperMultiGraph
    decompositions
    dem_to_decoding_graph_and_logicals
    dem_to_hypergraph_and_logicals
    DemParser
    DetectorCounter
    DetectorRecord
    DetectorRecorder
    EdgeRecord
    EdgeT
    errors_to_syndrome
    extract_logicals
    filter_to_data_edges
    filter_to_measure_edges
    FixedWidthBitstring
    get_round_words
    graph_to_json
    has_contiguous_nodes
    hypergraph_to_weighted_edge_list
    HyperLogicals
    HyperMultiGraph
    inverse_logical_at_boundary
    is_single_connected_component
    LogicalsInEdges
    NXCode
    NXDecodingGraph
    NXDecodingMultiGraph
    NXLogicals
    observable_warning
    OrderedDecodingEdges
    OrderedSyndrome
    parse_explained_dem
    single_boundary_is_last_node
    unweight_graph
    vector_weights
    worst_case_num_detectors

.. _api-deltakit-decode:

``deltakit.decode``
-------------------

.. currentmodule:: deltakit.decode

``deltakit.decode`` provides decoding algorithms for quantum error correction,
including minimum-weight perfect matching (MWPM), belief matching, and ambiguity clustering.

.. autosummary::
    :toctree: _build/generated/

    PyMatchingDecoder
    MWPMDecoder
    CCDecoder
    BeliefMatchingDecoder
    BPOSDecoder
    ACDecoder
    LCDecoder

.. _api-deltakit-decode-analysis:

``deltakit.decode.analysis``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.decode.analysis`` provides manager classes and utilities for running,
orchestrating, and benchmarking decoders on quantum circuits.

.. currentmodule:: deltakit.decode.analysis

.. autosummary::
    :toctree: _build/generated/

    DecoderManager
    InvalidGlobalManagerStateError
    run_decoding_on_circuit
    EmpiricalDecodingErrorDistribution
    B8DecoderManager
    GraphDecoderManager
    StimDecoderManager
    RunAllAnalysisEngine

.. _api-deltakit-decode-noise_sources:

``deltakit.decode.noise_sources``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.decode.noise_sources`` provides noise model definitions and sampling
tools used during decoding analysis.

.. currentmodule:: deltakit.decode.noise_sources

.. autosummary::
    :toctree: _build/generated/

    CombinedIndependent
    CombinedSequences
    MonteCarloNoise
    NoiseModel
    SequentialNoise
    EdgeProbabilityMatchingNoise
    ExhaustiveMatchingNoise
    ExhaustiveWeightedMatchingNoise
    FixedWeightMatchingNoise
    NoMatchingNoise
    UniformErasureNoise
    UniformMatchingNoise
    OptionedStim
    SampleStimNoise
    StimNoise
    ToyNoise

.. _api-deltakit-decode-utils:

``deltakit.decode.utils``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: deltakit.decode.utils

``deltakit.decode.utils`` provides utility functions for computing error probabilities,
building correlation matrices, and visualising 3D decoding graphs.

.. autosummary::
    :toctree: _build/generated/

    calculate_pij_values
    create_correlation_matrix
    create_dem_from_pij
    dem_and_pij_edges_max_diff
    generate_expectation_data
    make_logger
    parse_stim_circuit
    pij_and_dem_edge_diff
    pij_edges_max_diff
    pijs_edge_diff
    plot_correlation_matrix
    split_measurement_bitstring
    VisDecodingGraph3D

.. _api-deltakit-explorer:

``deltakit.explorer``
---------------------

.. currentmodule:: deltakit.explorer

``deltakit.explorer`` provides high-level client interfaces and orchestration
tools for managing QEC experiments.

.. autosummary::
    :toctree: _build/generated/

    Client
    Logging

.. _api-deltakit-explorer-analysis:

``deltakit.explorer.analysis``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.explorer.analysis`` provides functions for calculating logical error
probabilities (LEP), error suppression factors (Lambda), and fit parameters.

.. autosummary::
    :toctree: _build/generated/

    analysis.calculate_lambda_and_lambda_stddev
    analysis.calculate_lep_and_lep_stddev
    analysis.get_exp_fit
    analysis.get_lambda_fit

.. _api-deltakit-explorer-plotting:

``deltakit.explorer.plotting``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.explorer.plotting`` provides visualisation functions for defect diagrams,
defect rates, and correlation matrices.

.. autosummary::
    :toctree: _build/generated/

    plotting.correlation_matrix
    plotting.defect_diagram
    plotting.defect_rates

.. _api-deltakit-explorer-codes:

``deltakit.explorer.codes``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.explorer.codes`` provides quantum error correction code families
(such as planar, toric, repetition, and bivariate bicycle codes) and experiment circuit generators.

.. autosummary::
    :toctree: _build/generated/

    codes.CSSCode
    codes.css_code_memory_circuit
    codes.css_code_stability_circuit
    codes.CSSStage
    codes.experiment_circuit
    codes.StabiliserCode
    codes.BivariateBicycleCode
    codes.Monomial
    codes.Polynomial
    codes.ScheduleType
    codes.RotatedPlanarCode
    codes.UnrotatedPlanarCode
    codes.UnrotatedToricCode
    codes.RepetitionCode

.. _api-deltakit-explorer-data:

``deltakit.explorer.data``
^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.explorer.data`` provides I/O helpers for reading and writing experiment
data in binary and text formats (such as 01, b8, and CSV).

.. autosummary::
    :toctree: _build/generated/

    data.read_01
    data.read_b8
    data.read_csv
    data.write_01
    data.write_b8
    data.write_binary_data_to_file

.. _api-deltakit-explorer-enums:

``deltakit.explorer.enums``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.explorer.enums`` defines enumerations for data formats, decoder types,
error-correcting code families, and experiment types.

.. autosummary::
    :toctree: _build/generated/

    enums.DataFormat
    enums.DecoderType
    enums.QECECodeType
    enums.QECExperimentType

.. _api-deltakit-explorer-qpu:

``deltakit.explorer.qpu``
^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.explorer.qpu`` provides QPU hardware abstractions, native gate sets,
and realistic noise models.

.. autosummary::
    :toctree: _build/generated/

    qpu.QPU
    qpu.NativeGateSet
    qpu.NativeGateSetAndTimes
    qpu.ExhaustiveGateSet
    qpu.NoiseParameters
    qpu.PhenomenologicalNoise
    qpu.SD6Noise
    qpu.SI1000Noise
    qpu.ToyNoise
    qpu.PhysicalNoise

.. _api-deltakit-explorer-simulation:

``deltakit.explorer.simulation``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.explorer.simulation`` provides simulation wrappers for executing circuits
and generating measurement samples with Stim.

.. autosummary::
    :toctree: _build/generated/

    simulation.simulate_with_stim

.. _api-deltakit-explorer-types:

``deltakit.explorer.types``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``deltakit.explorer.types`` defines structured container types for measurements,
detection events, noise models, and experiment specifications.

.. autosummary::
    :toctree: _build/generated/

    types.DataString
    types.QECExperiment
    types.QECExperimentDefinition
    types.BinaryDataType
    types.CircuitParameters
    types.Decoder
    types.DecodingResult
    types.DetectionEvents
    types.LeakageFlags
    types.MatrixSpecifications
    types.Measurements
    types.NoiseModel
    types.ObservableFlips
    types.PhysicalNoiseModel
    types.QubitCoordinateToDetectorMapping
    types.RAMData
    types.SI1000NoiseModel
    types.Sizes
    types.TypedData
    types.TypedDataFile
    types.TypedDataString
