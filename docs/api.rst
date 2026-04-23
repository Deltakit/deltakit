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

Description of ``deltakit.circuit.noise_channels`` namespace here.

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

Description of ``deltakit.core`` namespace here.

.. autosummary::
    :toctree: _build/generated/

.. _api-deltakit-core-data_formats:

``deltakit.core.data_formats``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description of ``deltakit.core.data_formats`` namespace here.

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

Description of ``deltakit.core.decoding_graphs`` namespace here.

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

Description of ``deltakit.decode`` namespace here.

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

Description of ``deltakit.decode.analysis`` namespace here.

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

Description of ``deltakit.decode.noise_sources`` namespace here.

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

Description of ``deltakit.explorer`` namespace here.

.. autosummary::
    :toctree: _build/generated/

    Client
    Logging

.. _api-deltakit-explorer-analysis:

``deltakit.explorer.analysis``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description of ``deltakit.explorer.analysis`` namespace here.

.. autosummary::
    :toctree: _build/generated/

    analysis.calculate_lambda_and_lambda_stddev
    analysis.calculate_lep_and_lep_stddev
    analysis.get_exp_fit
    analysis.get_lambda_fit

.. _api-deltakit-explorer-plotting:

``deltakit.explorer.plotting``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description of ``deltakit.explorer.plotting`` namespace here.

.. autosummary::
    :toctree: _build/generated/

    plotting.correlation_matrix
    plotting.defect_diagram
    plotting.defect_rates

.. _api-deltakit-explorer-codes:

``deltakit.explorer.codes``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description of ``deltakit.explorer.codes`` namespace here.

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

Description of ``deltakit.explorer.data`` namespace here.

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

Description of ``deltakit.explorer.enums`` namespace here.

.. autosummary::
    :toctree: _build/generated/

    enums.DataFormat
    enums.DecoderType
    enums.QECECodeType
    enums.QECExperimentType

.. _api-deltakit-explorer-qpu:

``deltakit.explorer.qpu``
^^^^^^^^^^^^^^^^^^^^^^^^^

Description of ``deltakit.explorer.qpu`` namespace here.

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

Description of ``deltakit.explorer.simulation`` namespace here.

.. autosummary::
    :toctree: _build/generated/

    simulation.simulate_with_stim

.. _api-deltakit-explorer-types:

``deltakit.explorer.types``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description of ``deltakit.explorer.types`` namespace here.

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
