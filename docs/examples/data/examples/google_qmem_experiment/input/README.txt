This directory contains subdirectories each containing data files describing
quantum error correction experiments.

Each subdirectory's name follows the pattern
"{code}_b{basis}_d{distance}_r{rounds}_center_{row}_{col}".
The row and col are the coordinates of the center data qubit. The basis is
whether the X or Z logical observable was tested. The distance is the code
distance of the experiment. The number of rounds is how many times each
measurement qubit was measured (the data measurements at the end of the circuit
aren't counted as an additional round).

As an example of how these files can be used, the following bash command line
can be used to find the high energy event (noted in the paper) that increased
the logical error rate of the repetition code at distance 25:

        # Assumes a linux-like command line environment.
        # Assumes your working directory is the directory containing this readme.
        # Note the huge cluster of mismatches near shot 57775.
        paste -d '' \
            repetition_code_bZ_d25_r50_center_5_5/obs_flips_actual.01 \
            repetition_code_bZ_d25_r50_center_5_5/obs_flips_predicted_by_correlated_matching.01 \
            | grep -P "01|10" -n

Note: the repetition code subdirectory only includes the full repetition code
experiment data, not the subsampled experiments derived by truncating the data
or the postselected experiments derived by removing shots near the high energy
event. The repetition code subdirectory also doesn't include predictions from
the tensor contraction decoder or the belief matching decoder, as they weren't
used for the repetition code data.

The following is an inventory of the files that appear in each subdirectory.

properties.yml:
    Describes high level properties of the experiment, such as the number of
    rounds and the number of measurements in the circuit.

    Stored in YAML format ( see https://yaml.org/spec/1.2.2/ ).

layout.svg:
    A diagram of the stabilizers and observables defining the experiment, as
    well as their relative position on the chip.

    Stored in SVG format ( see https://www.w3.org/TR/SVG2/ ).
    Black squares are four body X stabilizers.
    Red squares are four body Z stabilizers.
    Black semicircles are two body X stabilizers.
    Red semicircles are two body Z stabilizers.
    Zig zag arrows indicate the order of operations within a stabilizer.
    Yellow circles indicate "pauses" where the order of operations waits a layer.
    Coordinate annotations indicate the positions of the qubits on the chip.
    X/Z annotations indicate the location of the logical observable being tested.

circuit_ideal.stim:
    The circuit that was sampled from, including annotations describing how
    detection events are computed and what the logical observable is. The
    circuit is needed in order to convert measurement data and sweep data into
    detection event data and observable flip data.

    The python package stimcirq can be used to convert this into a cirq circuit
    and then cirq can be used to convert it into other formats such as QASM.

    Stored in STIM circuit format ( see https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md ).

measurements.b8:
    The actual measurement data collected from hardware. See the accompanying
    circuit file or properties file to determine how many measurements there are
    per shot. Detection event data and observable flip data is derived from this
    data.

    Stored in b8 format ( see https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#b8 ).
    Bits per shot is circuit_measurements from properties.yml.
    Each shot's data is byte aligned by padding up to a multiple of 8 bits.
    Bits are packed into bytes in little endian order.

sweep.b8:
    Bit packed circuit configuration data, describing which sweep bits were set
    during each shot. Specifically, the sweep bits are used to initialize the
    data qubits into different patterns of 0s and 1s between shots. These bits
    determine whether instructions like `CX sweep[5] 5` in the circuit file are
    turned into X gates or I gates. See the circuit file or the properties file
    to determine how many sweep bits are used in the circuit. Detection event
    data and observable flip data is derived from this data.

    Stored in b8 format ( see https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#b8 ).
    Bits per shot is circuit_sweep_bits from properties.yml.
    Each shot's data is byte aligned by padding up to a multiple of 8 bits.
    Bits are packed into bytes in little endian order.

circuit_noisy.stim:
    Same as circuit_ideal.stim, except noise annotations have been added to the
    circuit. The noise annotations implement a Pauli noise model that
    approximates the noise which occurs on hardware.

    Stored in STIM circuit format ( see https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md ).

circuit_detector_error_model.dem:
    A detector error model derived directly from circuit_noisy.stim using stim.
    This file can be used to configure a decoder. It represents errors as
    edges in a weighted hypergraph, where nodes correspond to possible detection
    events. Errors with more than two detection events also contain suggested
    decompositions into graphlike errors (errors with at most two detection
    events).

    This file can be rederived from circuit_noisy.stim using `stim analyze_errors`:

        # Assumes a linux-like command line environment.
        # Assumes a python environment with stim 1.9+ installed.
        # Assumes your working directory is the relevant experiment directory.
        # The regenerated file may differ due to variation in float rounding.
        stim analyze_errors \
            --decompose_errors \
            --fold_loops \
            --in circuit_noisy.stim \
            --out circuit_detector_error_model.dem

    Stored in DEM format ( see https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md ).

pij_from_even_for_odd.dem:
    A detector error model derived from experimental data. Error mechanisms have
    been reweighted based on how often associated detection events are seen to
    actually occur.

    This model was derived from the detection event data from shots that have an
    EVEN index. (Note that the first shot has index 0, not index 1.) In principle
    it should be safe to use this model when decoding all shots, because it is
    created without using the observable flip data that actually needs to be
    predicted (so it's difficult to overfit the model, even intentionally).
    However, to be cautious, we recommend only using it on odd index shots.

pij_from_odd_for_even.dem:
    A detector error model derived from experimental data. Error mechanisms have
    been reweighted based on how often associated detection events are seen to
    actually occur.

    This model was derived from the detection event data from shots that have an
    ODD index. (Note that the first shot has index 0, not index 1.) In principle
    it should be safe to use this model when decoding all shots, because it is
    created without using the observable flip data that actually needs to be
    predicted (so it's difficult to overfit the model, even intentionally).
    However, to be cautious, we recommend only using it on even index shots.

detection_events.b8:
    Bit packed detection event data, describing which detectors fired and did
    not fire, from all of the shots that were performed. See the circuit file or
    the properties file to determine how many detectors are in the circuit. The
    detection event data is used by decoders to predict whether or not the
    logical observable was flipped.

    This file can be rederived from circuit_ideal.stim, measurements.b8, and
    sweep.b8 using `stim m2d`:

        # Assumes a linux-like command line environment.
        # Assumes a python environment with stim 1.9+ installed.
        # Assumes your working directory is the relevant experiment directory.
        # Generates BOTH detection_events.b8 and obs_flips_actual.01
        stim m2d \
            --circuit circuit_ideal.stim \
            --in measurements.b8 \
            --in_format b8 \
            --sweep sweep.b8 \
            --sweep_format b8 \
            --out detection_events.b8 \
            --out_format b8 \
            --obs_out obs_flips_actual.01 \
            --obs_out_format 01

    Stored in b8 format ( see https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#b8 ).
    Bits per shot is circuit_detectors from properties.yml.
    Each shot's data is byte aligned by padding up to a multiple of 8 bits.
    Bits are packed into bytes in little endian order.

obs_flips_actual.b8:
    Observable flip data, describing which observables were flipped or not in
    each shot. Note that the stored value is *NOT* the actual value of the
    observable but rather whether it was flipped compared to what the value
    would have been if the circuit had executed noiselessly. Although circuits
    can declare multiple observables, all experiments we did had just one
    checked observable. This data is the data that decoders are supposed to
    predict. It is the ground truth decoders are measured against.

    This file can be rederived from circuit_ideal.stim, measurements.b8, and
    sweep.b8 using `stim m2d`:

        # Assumes a linux-like command line environment.
        # Assumes a python environment with stim 1.9+ installed.
        # Assumes your working directory is the relevant experiment directory.
        # Generates BOTH detection_events.b8 and obs_flips_actual.01
        stim m2d \
            --circuit circuit_ideal.stim \
            --in measurements.b8 \
            --in_format b8 \
            --sweep sweep.b8 \
            --sweep_format b8 \
            --out detection_events.b8 \
            --out_format b8 \
            --obs_out obs_flips_actual.01 \
            --obs_out_format 01

    Stored in 01 format ( see https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#01 ).
    Each line in the file is one shot, with the line '0' meaning 'observable
    was not flipped' (i.e. the observed result agreed with the result that
    would occur if there was no noise) and the line '1' meaning 'observable
    was flipped'.

obs_flips_predicted_by_pymatching.01:
    The predictions produced by PyMatching (see https://arxiv.org/abs/2105.13082
    and https://github.com/oscarhiggott/PyMatching ). PyMatching is a minimum
    weight perfect matching decoder.

    NOTE: The decoder was configured using circuit_detector_error_model.dem for
    all shots.

    The number of mistakes made by this decoder can be determined by counting
    mismatches versus obs_flips_actual.b8:

        # Assumes a linux-like command line environment.
        # Assumes your working directory is the relevant experiment directory.
        paste -d '' \
            obs_flips_actual.01 \
            obs_flips_predicted_by_pymatching.01 \
            | grep -P "01|10" \
            | wc -l

    Because PyMatching is open source, these predictions can be rederived from
    the detection event data and a detector error model. The easiest way to do
    this is by using the python package "sinter" to feed the provided files to
    pymatching:

        # Assumes a linux-like command line environment.
        # Assumes a python environment with sinter 1.9+ installed.
        # Assumes your working directory is the relevant experiment directory.
        # Note: takes minutes or even tens of minutes to finish.
        sinter predict \
            --decoder pymatching \
            --dem circuit_detector_error_model.dem \
            --dets detection_events.b8 \
            --dets_format b8 \
            --obs_out obs_flips_predicted_by_pymatching.01 \
            --obs_out_format 01

    Stored in 01 format ( see https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#01 ).
    Each line in the file is one shot, with the line '0' meaning 'I predict
    the observable was not flipped' and the line '1' meaning 'I predict the
    observable was flipped'.

obs_flips_predicted_by_correlated_matching.01:
    The predictions produced by an internal decoder which performs minimum
    weight perfect matching after adjusting edge weights using heuristics that
    account for correlations between the X and Z decoding graphs.

    NOTE: The decoder was configured using circuit_detector_error_model.dem for
    all shots.

    The number of mistakes made by this decoder can be determined by counting
    mismatches versus obs_flips_actual.b8:

        # Assumes a linux-like command line environment.
        # Assumes your working directory is the relevant experiment directory.
        paste -d '' \
            obs_flips_actual.01 \
            obs_flips_predicted_by_correlated_matching.01 \
            | grep -P "01|10" \
            | wc -l

    Stored in 01 format ( see https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#01 ).
    Each line in the file is one shot, with the line '0' meaning 'I predict
    the observable was not flipped' and the line '1' meaning 'I predict the
    observable was flipped'.

obs_flips_predicted_by_belief_matching.01:
    The predictions produced by an internal decoder which combines belief
    propagation and minimum weight perfect matching, similar to the approach
    described in https://arxiv.org/abs/2203.04948 . See the paper
    supplement for details.

    NOTE: For odd numbered shots the decoder was configured using
    pij_from_even_for_odd.dem while for even numbered shots it was configured
    using pij_from_odd_for_even.dem.

    The number of mistakes made by this decoder can be determined by counting
    mismatches versus obs_flips_actual.b8:

        # Assumes a linux-like command line environment.
        # Assumes your working directory is the relevant experiment directory.
        paste -d '' \
            obs_flips_actual.01 \
            obs_flips_predicted_by_belief_matching.01 \
            | grep -P "01|10" \
            | wc -l

    Stored in 01 format ( see https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#01 ).
    Each line in the file is one shot, with the line '0' meaning 'I predict
    the observable was not flipped' and the line '1' meaning 'I predict the
    observable was flipped'.

obs_flips_predicted_by_tensor_network_contraction.01:
    The predictions produced by an internal decoder, using tensor network
    contraction to approximate maximum-likelihood decoding. See the paper
    supplement for details.

    NOTE: For odd numbered shots the decoder was configured using
    pij_from_even_for_odd.dem while for even numbered shots it was configured
    using pij_from_odd_for_even.dem.

    The number of mistakes made by this decoder can be determined by counting
    mismatches versus obs_flips_actual.b8:

        # Assumes a linux-like command line environment.
        # Assumes your working directory is the relevant experiment directory.
        paste -d '' \
            obs_flips_actual.01 \
            obs_flips_predicted_by_tensor_network_contraction.01 \
            | grep -P "01|10" \
            | wc -l

    Stored in 01 format ( see https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md#01 ).
    Each line in the file is one shot, with the line '0' meaning 'I predict
    the observable was not flipped' and the line '1' meaning 'I predict the
    observable was flipped'.

