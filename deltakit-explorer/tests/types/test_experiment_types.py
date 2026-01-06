import pytest

from deltakit_explorer.types._experiment_types import (
    PauliBasis,
    QECECodeType,
    QECExperiment,
    QECExperimentDefinition,
    QECExperimentType,
)


def test_get_parameters_gql_string_none():
    exp = QECExperimentDefinition(
        experiment_type=QECExperimentType.QUANTUM_MEMORY,
        code_type=QECECodeType.REPETITION,
        observable_basis=PauliBasis.Z,
        num_rounds=1,
        basis_gates=None,
        parameters=None
    )
    assert exp.get_parameters_gql_string() is None

def test_get_repetition_z_quantum_memory():
    exp = QECExperimentDefinition.get_repetition_z_quantum_memory(3, 2)
    assert exp.code_type == QECECodeType.REPETITION

def test_from_circuit_and_measurements_negative(tmp_path):
    stim_path = tmp_path / "circuit.stim"
    stim_path.write_text("M 0\n")
    meas_path = tmp_path / "meas.b8"
    meas_path.write_text("01\n")
    with pytest.raises(ValueError):
        QECExperiment.from_circuit_and_measurements(stim_path, meas_path, "B8", sweep_path=tmp_path/"sweep", sweep_format=None)
