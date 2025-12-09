from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import ClassVar

import numpy
import numpy.typing as npt
import pytest
from deltakit_circuit import Circuit
from deltakit_circuit._noise_factory import NoiseProfile
from deltakit_circuit.noise_channels._depolarising_noise import Depolarise1
from typing_extensions import override

from deltakit_explorer.analysis.budget import NoiseInterface
from deltakit_explorer.analysis.budget._generation import (
    _generate_surface_code_memory_decoder_manager,
    generate_decoder_managers_for_lambda,
)
from deltakit_explorer.analysis.budget._memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.qpu._noise._noise_parameters import NoiseParameters
from deltakit_explorer.qpu._qpu import QPU


def get_noise_model_type(num_parameters: int) -> type[NoiseInterface]:
    class _NoiseModel(NoiseInterface[Circuit]):
        num_noise_parameters: ClassVar[int] = num_parameters
        parameter_names: ClassVar[tuple[str, ...]] = tuple(
            f"param{i}" for i in range(num_parameters)
        )

        def __init__(
            self,
            noise_parameters: Sequence[float] | npt.NDArray[numpy.floating],
            name: str | None = "simple",
        ):
            super().__init__(noise_parameters, name)

        @override
        def apply(self, computation: Circuit) -> Circuit:
            gate_noise: list[NoiseProfile] = [
                lambda noise_context: Depolarise1.generator_from_prob(
                    self._noise_parameters[0]
                )(noise_context.gate_layer_qubits(None, gate_qubit_count=1))
            ]
            qpu = QPU(
                computation.qubits, noise_model=NoiseParameters(gate_noise=gate_noise)
            )
            return qpu.compile_and_add_noise_to_circuit(computation)

        @override
        @classmethod
        def is_valid(cls, parameters: npt.NDArray[numpy.floating]) -> str | None:
            return None

    return _NoiseModel


@pytest.mark.parametrize(
    "noise_model,memgen",
    itertools.product(
        [
            get_noise_model_type(1)([0.1]),
            get_noise_model_type(2)([0.1, 0.2]),
            get_noise_model_type(100)(0.25 + numpy.arange(100) / 200),
        ],
        [get_rotated_surface_code_memory_circuit],
    ),
)
def test_decoder_manager_has_metadata(
    noise_model: NoiseInterface, memgen: MemoryGenerator
) -> None:
    dm = _generate_surface_code_memory_decoder_manager(3, 3, noise_model, memgen)
    assert dm is not None
    assert "distance" in dm.metadata
    assert isinstance(dm.metadata["distance"], int)
    assert "num_rounds" in dm.metadata
    assert isinstance(dm.metadata["num_rounds"], int)
    for noise_name in noise_model.parameter_names:
        assert f"noise_{noise_name}" in dm.metadata
        assert isinstance(dm.metadata[f"noise_{noise_name}"], float)


@pytest.mark.parametrize("n,m", [(1, 10), (11, 10)])
class TestGenerateDecoderManagerForLambda:
    @pytest.fixture
    def xis(
        self, random_generator: numpy.random.Generator, n: int, m: int
    ) -> npt.NDArray[numpy.floating]:
        return random_generator.random((n, m)) * 0.1

    @pytest.fixture
    def noise_model(self, n: int) -> NoiseInterface:
        return get_noise_model_type(n)([0.01 for _ in range(n)])

    def test_generate_decoder_managers_for_lambda(
        self, xis: npt.NDArray[numpy.floating], noise_model: NoiseInterface
    ) -> None:
        n, m = xis.shape
        dms = generate_decoder_managers_for_lambda(
            xis, type(noise_model), {3: [6], 5: [6]}
        )
        assert len(dms) == 2 * m
        for dm in dms:
            assert "distance" in dm.metadata
            assert isinstance(dm.metadata["distance"], int)
            assert "num_rounds" in dm.metadata
            assert isinstance(dm.metadata["num_rounds"], int)
            for noise_name in noise_model.parameter_names:
                assert f"noise_{noise_name}" in dm.metadata
                assert isinstance(dm.metadata[f"noise_{noise_name}"], float)
