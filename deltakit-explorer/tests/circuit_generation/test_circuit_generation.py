import pytest
import stim
from deltakit_circuit.gates import PauliBasis
from deltakit_explorer.codes._bivariate_bicycle_code import \
    BivariateBicycleCode  # noqa: F401
from deltakit_explorer.codes._css._css_code_experiment_circuit import (
    css_code_memory_circuit, css_code_stability_circuit)
from deltakit_explorer.codes._planar_code._rotated_planar_code import \
    RotatedPlanarCode
from deltakit_explorer.codes._planar_code._unrotated_planar_code import \
    UnrotatedPlanarCode
from deltakit_explorer.codes._planar_code._unrotated_toric_code import \
    UnrotatedToricCode
from deltakit_explorer.codes._repetition_code import RepetitionCode


@pytest.fixture
def mock_client(mocker):
    client = mocker.Mock()
    # Return a minimal valid stim circuit string
    client.generate_circuit.return_value = "H 0\nTICK\nM 0"
    return client


@pytest.mark.parametrize("code", [
    RotatedPlanarCode(width=3, height=3),
    UnrotatedPlanarCode(width=3, height=3),
    UnrotatedToricCode(3, 3),
    RepetitionCode(distance=3),
])
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
