import deltakit_stim as stim
import numpy as np
import pytest
from deltakit_circuit.gates import MX, PauliBasis

from deltakit_explorer.codes._bivariate_bicycle_code import BivariateBicycleCode
from deltakit_explorer.codes._css._check_redundancies import (
    build_check_redundancy_detectors,
    check_matrix_for_memory_basis,
    check_redundancy_record_sets,
    detector_annotation_rank,
    detector_parity_matrix,
    first_syndrome_round_measurement_records,
    left_nullspace_basis,
    measurement_gate_index,
    stabiliser_row_indices_for_memory_basis,
)
from deltakit_explorer.codes._css._css_code_experiment_circuit import (
    css_code_memory_circuit,
    css_code_stability_circuit,
)
from deltakit_explorer.codes._planar_code._rotated_planar_code import RotatedPlanarCode
from deltakit_explorer.codes._planar_code._unrotated_planar_code import (
    UnrotatedPlanarCode,
)
from deltakit_explorer.codes._planar_code._unrotated_toric_code import (
    UnrotatedToricCode,
)
from deltakit_explorer.codes._repetition_code import RepetitionCode
from tests.helpers._gf2 import quotient_dimension_gf2


def _count_detectors(circuit) -> int:
    return len(circuit.detectors(include_nested=True))


def _count_check_redundancy_detectors(circuit) -> int:
    return sum(
        1
        for detector in circuit.detectors(include_nested=True)
        if detector.tag == "check_redundancy"
    )


def _meta_check_detectors(circuit):
    return [
        detector
        for detector in circuit.detectors(include_nested=True)
        if detector.tag == "check_redundancy"
    ]


@pytest.fixture
def bb_72_12_6_code():
    return BivariateBicycleCode(
        param_l=6,
        param_m=6,
        m_A_powers=[3, 1, 2],
        m_B_powers=[3, 1, 2],
    )


@pytest.mark.parametrize(
    ("param_l", "param_m", "m_A_powers", "m_B_powers", "expected_meta"),
    [
        (6, 6, [3, 1, 2], [3, 1, 2], 6),
        (15, 3, [9, 1, 2], [0, 2, 7], 4),
        (9, 6, [3, 1, 2], [3, 1, 2], 4),
    ],
)
@pytest.mark.parametrize("num_rounds", [1, 2, 3])
@pytest.mark.parametrize("basis", [PauliBasis.Z, PauliBasis.X])
def test_bb_check_redundancy_opt_in(
    param_l, param_m, m_A_powers, m_B_powers, expected_meta, num_rounds, basis
):
    code = BivariateBicycleCode(
        param_l=param_l,
        param_m=param_m,
        m_A_powers=m_A_powers,
        m_B_powers=m_B_powers,
    )
    default_circuit = css_code_memory_circuit(
        code, num_rounds=num_rounds, logical_basis=basis
    )
    opt_in_circuit = css_code_memory_circuit(
        code,
        num_rounds=num_rounds,
        logical_basis=basis,
        include_check_redundancies=True,
    )
    assert _count_detectors(default_circuit) + expected_meta == _count_detectors(
        opt_in_circuit
    )
    assert _count_check_redundancy_detectors(opt_in_circuit) == expected_meta
    assert _count_check_redundancy_detectors(default_circuit) == 0


def test_planar_code_default_unchanged_with_opt_in_flag():
    code = RotatedPlanarCode(width=3, height=3)
    default_circuit = css_code_memory_circuit(code, 3, PauliBasis.Z)
    opt_in_circuit = css_code_memory_circuit(
        code, 3, PauliBasis.Z, include_check_redundancies=True
    )
    assert default_circuit.as_stim_circuit() == opt_in_circuit.as_stim_circuit()


def test_left_nullspace_toy_dependency():
    H = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    basis = left_nullspace_basis(H)
    assert basis.shape == (1, 4)
    assert (basis[0] == np.array([1, 1, 1, 0], dtype=np.uint8)).all()


def test_bb_meta_detector_fan_in(bb_72_12_6_code):
    code = bb_72_12_6_code
    circuit = css_code_memory_circuit(code, 3, PauliBasis.Z)
    meta = build_check_redundancy_detectors(code, circuit, PauliBasis.Z, 3)
    assert len(meta) == 6
    fan_ins = [len(detector.measurements) for detector in meta]
    assert min(fan_ins) >= 16
    assert max(fan_ins) <= 18


def test_bb_meta_checks_use_first_round_mx_records(bb_72_12_6_code):
    code = bb_72_12_6_code
    circuit = css_code_memory_circuit(code, 3, PauliBasis.Z)
    measurement_gates = circuit.measurement_gates
    num_measurements = len(measurement_gates)
    meta = build_check_redundancy_detectors(code, circuit, PauliBasis.Z, 3)
    for detector in meta:
        for record in detector.measurements:
            gate = measurement_gates[measurement_gate_index(num_measurements, record)]
            assert isinstance(gate, MX)


def test_bb_check_matrix_side_for_z_memory(bb_72_12_6_code):
    code = bb_72_12_6_code
    hx = check_matrix_for_memory_basis(code, PauliBasis.Z, random_basis_side=True)
    hz = check_matrix_for_memory_basis(code, PauliBasis.Z, random_basis_side=False)
    np.testing.assert_array_equal(hx, code.m_Hx)
    np.testing.assert_array_equal(hz, code.m_Hz)
    assert not np.array_equal(hx, hz)


def test_bb_meta_checks_increase_annotation_rank(bb_72_12_6_code):
    code = bb_72_12_6_code
    default_circuit = css_code_memory_circuit(code, 3, PauliBasis.Z)
    opt_in_circuit = css_code_memory_circuit(
        code, 3, PauliBasis.Z, include_check_redundancies=True
    )
    assert (
        detector_annotation_rank(opt_in_circuit)
        == detector_annotation_rank(default_circuit) + 6
    )


def test_bb_meta_checks_outside_default_detector_span(bb_72_12_6_code):
    code = bb_72_12_6_code
    default_circuit = css_code_memory_circuit(code, 3, PauliBasis.Z)
    opt_in_circuit = css_code_memory_circuit(
        code, 3, PauliBasis.Z, include_check_redundancies=True
    )
    num_measurements = len(default_circuit.measurement_gates)
    default_matrix = detector_parity_matrix(
        default_circuit.detectors(include_nested=True), num_measurements
    )
    meta_matrix = detector_parity_matrix(
        _meta_check_detectors(opt_in_circuit), num_measurements
    )
    assert quotient_dimension_gf2(meta_matrix, default_matrix) == 6


def test_bb_wrong_side_record_sets_differ_from_correct(bb_72_12_6_code):
    code = bb_72_12_6_code
    circuit = css_code_memory_circuit(code, 3, PauliBasis.Z)
    correct_sets = check_redundancy_record_sets(code, circuit, PauliBasis.Z, 3)
    wrong_sets = tuple(
        frozenset(detector.measurements)
        for detector in build_check_redundancy_detectors(
            code, circuit, PauliBasis.Z, 3, random_basis_side=False
        )
    )
    assert len(wrong_sets) == 6
    assert wrong_sets != correct_sets


def test_bb_hz_matrix_with_x_records_differs_from_correct_meta(bb_72_12_6_code):
    code = bb_72_12_6_code
    circuit = css_code_memory_circuit(code, 3, PauliBasis.Z)
    correct_sets = check_redundancy_record_sets(code, circuit, PauliBasis.Z, 3)
    stage = code.measure_stabilisers(num_rounds=1)
    x_rows = stabiliser_row_indices_for_memory_basis(
        stage, PauliBasis.Z, random_basis_side=True
    )
    hz = check_matrix_for_memory_basis(code, PauliBasis.Z, random_basis_side=False)
    records = first_syndrome_round_measurement_records(
        circuit, len(stage.ordered_stabilisers)
    )
    stabiliser_to_record = {
        index: records[index] for index in range(len(stage.ordered_stabilisers))
    }
    side_bug_sets = []
    for dependency in left_nullspace_basis(hz):
        participating = [
            x_rows[row_index] for row_index, bit in enumerate(dependency) if bit
        ]
        if len(participating) >= 2:
            side_bug_sets.append(
                frozenset(stabiliser_to_record[index] for index in participating)
            )
    assert len(side_bug_sets) == 6
    assert side_bug_sets != list(correct_sets)


def test_bb_opt_in_detector_error_model_compiles(bb_72_12_6_code):
    code = bb_72_12_6_code
    opt_in_circuit = css_code_memory_circuit(
        code, 3, PauliBasis.Z, include_check_redundancies=True
    )
    opt_in_circuit.as_stim_circuit().detector_error_model(decompose_errors=True)


def test_bb_check_redundancy_record_sets_accessor(bb_72_12_6_code):
    code = bb_72_12_6_code
    circuit = css_code_memory_circuit(code, 3, PauliBasis.Z)
    record_sets = code.check_redundancy_record_sets(circuit, PauliBasis.Z, 3)
    assert record_sets == check_redundancy_record_sets(code, circuit, PauliBasis.Z, 3)
    assert len(record_sets) == 6
    built = build_check_redundancy_detectors(code, circuit, PauliBasis.Z, 3)
    assert record_sets == tuple(frozenset(detector.measurements) for detector in built)


@pytest.fixture
def mock_client(mocker):
    client = mocker.Mock()
    # Return a minimal valid stim circuit string
    client.generate_circuit.return_value = "H 0\nTICK\nM 0"
    return client


@pytest.mark.parametrize(
    "code",
    [
        RotatedPlanarCode(width=3, height=3),
        UnrotatedPlanarCode(width=3, height=3),
        UnrotatedToricCode(3, 3),
        RepetitionCode(distance=3),
    ],
)
@pytest.mark.parametrize("basis", [PauliBasis.X, PauliBasis.Z])
def test_css_code_experiment_circuit_cloud(mock_client, code, basis):
    result = css_code_memory_circuit(
        css_code=code,
        num_rounds=3,
        logical_basis=basis,
        client=mock_client,
    )
    assert result.as_stim_circuit() == stim.Circuit("H 0\nTICK\nM 0")
    mock_client.generate_circuit.assert_called_once()


def test_cloud_css_code_experiment_circuit_no_client():
    code = RotatedPlanarCode(width=3, height=3)
    with pytest.raises(NotImplementedError, match="A `client` is required"):
        css_code_stability_circuit(code, 2, PauliBasis.X, client=None)
