from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from deltakit_circuit import Circuit
from deltakit_circuit._noise_factory import NoiseProfile
from deltakit_circuit.noise_channels._depolarising_noise import Depolarise1

from deltakit_explorer.analysis.error_budget._generation import (
    _generate_surface_code_memory_decoder_manager,
    generate_decoder_managers_for_lambda,
)
from deltakit_explorer.analysis.error_budget._memory import (
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.qpu._noise._noise_parameters import NoiseParameters
from deltakit_explorer.qpu._qpu import QPU


def noise_model(computation: Circuit, parameters: npt.NDArray[np.floating]) -> Circuit:
    gate_noise: list[NoiseProfile] = [
        lambda noise_context: Depolarise1.generator_from_prob(parameters[0])(
            noise_context.gate_layer_qubits(None, gate_qubit_count=1)
        )
    ]
    qpu = QPU(computation.qubits, noise_model=NoiseParameters(gate_noise=gate_noise))
    return qpu.compile_and_add_noise_to_circuit(computation)


def test_decoder_manager_has_metadata() -> None:
    parameters = np.array([0.1, 0.2])
    parameter_names = ["param0", "dfkjn"]
    dm = _generate_surface_code_memory_decoder_manager(
        3,
        3,
        noise_model,
        parameters,
        get_rotated_surface_code_memory_circuit,
        parameter_names,
    )
    assert dm is not None
    assert "distance" in dm.metadata
    assert isinstance(dm.metadata["distance"], int)
    assert "num_rounds" in dm.metadata
    assert isinstance(dm.metadata["num_rounds"], int)
    for noise_name in parameter_names:
        assert f"noise_{noise_name}" in dm.metadata
        assert isinstance(dm.metadata[f"noise_{noise_name}"], float)


@pytest.mark.parametrize(("n", "m"), [(1, 10), (11, 10)])
class TestGenerateDecoderManagerForLambda:
    @pytest.fixture
    def xis(
        self, random_generator: np.random.Generator, n: int, m: int
    ) -> npt.NDArray[np.floating]:
        return random_generator.random((m, n)) * 0.1

    def test_generate_decoder_managers_for_lambda(
        self, xis: npt.NDArray[np.floating]
    ) -> None:
        m, n = xis.shape
        parameter_names = [f"param{i}" for i in range(m)]
        dms = generate_decoder_managers_for_lambda(
            xis, noise_model, {3: [6], 5: [6]}, noise_parameter_names=parameter_names
        )
        assert len(dms) == 2 * n
        for dm in dms:
            assert "distance" in dm.metadata
            assert isinstance(dm.metadata["distance"], int)
            assert "num_rounds" in dm.metadata
            assert isinstance(dm.metadata["num_rounds"], int)
            for noise_name in parameter_names:
                assert f"noise_{noise_name}" in dm.metadata
                assert isinstance(dm.metadata[f"noise_{noise_name}"], float)
