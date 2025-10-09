"""Implementation of the noise interface using deltakit noise pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from numbers import Number
from typing import ClassVar, cast

from deltakit_circuit._noise_factory import NoiseProfile
import numpy
import numpy.typing as npt
from deltakit_circuit import Circuit, NoiseContext, measurement_noise_profile
from deltakit_circuit.gates._abstract_gates import (
    OneQubitCliffordGate,
    OneQubitMeasurementGate,
    OneQubitResetGate,
)
from deltakit_circuit.gates._measurement_gates import MEASUREMENT_GATES
from deltakit_circuit.gates._one_qubit_gates import ONE_QUBIT_GATES, I
from deltakit_circuit.gates._reset_gates import RESET_GATES
from deltakit_circuit.gates._two_qubit_gates import TWO_QUBIT_GATES
from deltakit_circuit.noise_channels._depolarising_noise import Depolarise1, Depolarise2
from typing_extensions import override

from deltakit_explorer.analysis.budget.interfaces import NoiseInterface
from deltakit_explorer.qpu import QPU, NativeGateSetAndTimes
from deltakit_explorer.qpu._noise._noise_parameters import (
    NoiseParameters,
    _idle_noise_from_t1_t2,
)
from deltakit_explorer.types._types import PhysicalNoiseModel


def physical_noise_model_to_noise_parameters_and_native_gate_set_and_times(
    noise_data: PhysicalNoiseModel,
) -> tuple[NoiseParameters, NativeGateSetAndTimes]:
    idle_noise = _idle_noise_from_t1_t2(noise_data.t_1, noise_data.t_2)

    def _gate_noise(noise_context: NoiseContext) -> list[Depolarise1 | Depolarise2]:
        noise_ops: list[Depolarise1 | Depolarise2] = []
        for gate in noise_context.gate_layer.gates:
            qubits = gate.qubits
            if len(qubits) == 2:
                noise_ops.append(
                    Depolarise2(qubits[0], qubits[1], noise_data.p_2_qubit_gate_error)
                )
            elif isinstance(gate, OneQubitCliffordGate) and not isinstance(gate, I):
                noise_ops.append(
                    Depolarise1(qubits[0], noise_data.p_1_qubit_gate_error)
                )
        return noise_ops

    gate_noise: list[NoiseProfile] = [_gate_noise]
    reset_noise: list[NoiseProfile] = [
        lambda noise_context: Depolarise1.generator_from_prob(noise_data.p_reset_error)(
            noise_context.gate_layer_qubits(OneQubitResetGate)
        )
    ]
    measurement_noise: list[NoiseProfile] = [
        lambda noise_context: Depolarise1.generator_from_prob(
            noise_data.p_meas_qubit_error
        )(noise_context.gate_layer_qubits(OneQubitMeasurementGate))
    ]
    measurement_flip_noise = measurement_noise_profile(noise_data.p_readout_flip)

    noise_parameters = NoiseParameters(
        gate_noise=gate_noise,
        idle_noise=idle_noise,
        reset_noise=reset_noise,
        measurement_noise_after=measurement_noise,
        measurement_flip=measurement_flip_noise,
    )

    native_gates = NativeGateSetAndTimes(
        one_qubit_gates={g: noise_data.time_1_qubit_gate for g in ONE_QUBIT_GATES},
        two_qubit_gates={g: noise_data.time_2_qubit_gate for g in TWO_QUBIT_GATES},
        reset_gates={r: noise_data.time_reset for r in RESET_GATES},
        measurement_gates={m: noise_data.time_measurement for m in MEASUREMENT_GATES},
    )
    return noise_parameters, native_gates


class DeltakitNoise(NoiseInterface[Circuit]):
    num_noise_parameters: ClassVar[int] = 11
    parameter_names: ClassVar[tuple[str, ...]] = (
        "T1",
        "T2",
        "1q gate time",
        "2q gate time",
        "Measurement time",
        "Reset time",
        "1q gate error",
        "2q gate error",
        "Reset error",
        "Measurement error",
        "Readout flip",
    )

    def __init__(
        self,
        noise_parameters: Sequence[float] | npt.NDArray[numpy.float64],
        name: str | None = None,
    ):
        super().__init__(noise_parameters, name)

    @classmethod
    def from_noise_parameters(
        cls,
        t_1: float = 0,
        t_2: float = 0,
        time_1_qubit_gate: float = 0,
        time_2_qubit_gate: float = 0,
        time_measurement: float = 0,
        time_reset: float = 0,
        p_1_qubit_gate_error: float = 0,
        p_2_qubit_gate_error: float = 0,
        p_reset_error: float = 0,
        p_meas_qubit_error: float = 0,
        p_readout_flip: float = 0,
        name: str | None = None,
    ) -> DeltakitNoise:
        return cls(
            [
                t_1,
                t_2,
                time_1_qubit_gate,
                time_2_qubit_gate,
                time_measurement,
                time_reset,
                p_1_qubit_gate_error,
                p_2_qubit_gate_error,
                p_reset_error,
                p_meas_qubit_error,
                p_readout_flip,
            ],
            name,
        )

    @staticmethod
    def from_physical_noise_model(
        noise_model: PhysicalNoiseModel, name: str | None = None
    ) -> DeltakitNoise:
        return DeltakitNoise.from_attributes_values(
            cast(Mapping[str, float], asdict(noise_model)), name
        )

    @staticmethod
    def from_attributes_values(
        noise_attributes: Mapping[str, float], name: str | None = None
    ) -> DeltakitNoise:
        strengths: list[float] = [0 for _ in DeltakitNoise.parameter_names]
        for noise_name, noise_strength in noise_attributes.items():
            assert isinstance(noise_strength, Number), type(noise_strength).__name__
            if index := DeltakitNoise.parameter_names.index(noise_name) == -1:
                raise ValueError(f"Invalid parameter name '{noise_name}'.")
            strengths[index] = noise_strength
        return DeltakitNoise(strengths, name)

    def _to_physical_noise_model(self) -> PhysicalNoiseModel:
        return PhysicalNoiseModel(
            t_1=self._get_value("T1"),
            t_2=self._get_value("T2"),
            time_1_qubit_gate=self._get_value("1q gate time"),
            time_2_qubit_gate=self._get_value("2q gate time"),
            time_measurement=self._get_value("Measurement time"),
            time_reset=self._get_value("Reset time"),
            p_1_qubit_gate_error=self._get_value("1q gate error"),
            p_2_qubit_gate_error=self._get_value("2q gate error"),
            p_reset_error=self._get_value("Reset error"),
            p_meas_qubit_error=self._get_value("Measurement error"),
            p_readout_flip=self._get_value("Readout flip"),
        )

    @override
    def apply(self, computation: Circuit) -> Circuit:
        noise_parameters, native_gates_times = (
            physical_noise_model_to_noise_parameters_and_native_gate_set_and_times(
                self._to_physical_noise_model()
            )
        )
        qpu = QPU(computation.qubits, native_gates_times, noise_parameters)
        return qpu.compile_and_add_noise_to_circuit(computation)

    @staticmethod
    def _check_is_positive(v: float, name: str) -> None:
        if v < 0:
            raise ValueError(
                f"{name.capitalize()} is expected to be positive but got {v}."
            )

    @staticmethod
    def _check_is_probability(v: float, name: str) -> None:
        if v < 0 or v > 1:
            raise ValueError(f"Expected 0 <= {name.capitalize()} <= 1 but got {v}.")

    @override
    @classmethod
    def is_valid(cls, parameters: npt.NDArray[numpy.float64]) -> str | None:
        t1, t2 = parameters[0], parameters[1]
        if t2 > 2 * t1:
            return f"Expected t2 ({t2:.3g}) <= 2 * t1 (2 * {t1:.3g} = {2 * t1:.3g})."
        for i in range(7):
            DeltakitNoise._check_is_positive(
                parameters[i], DeltakitNoise.parameter_names[i]
            )
        for i in range(7, 11):
            DeltakitNoise._check_is_probability(
                parameters[i], DeltakitNoise.parameter_names[i]
            )
        return super().is_valid(parameters)


class SimplerNoise(NoiseInterface):
    num_noise_parameters: ClassVar[int] = 2
    parameter_names: ClassVar[tuple[str, ...]] = (
        "Gate error",
        "Collapsing operation error",
    )

    def __init__(
        self,
        noise_parameters: Sequence[float] | npt.NDArray[numpy.float64],
        name: str | None = None,
    ):
        super().__init__(noise_parameters, name)

    @classmethod
    def from_noise_parameters(
        cls,
        gate_error: float = 0,
        collapsing_operation_error: float = 0,
        name: str | None = None,
    ) -> SimplerNoise:
        return cls([gate_error, collapsing_operation_error], name)

    def _to_physical_noise_model(self) -> PhysicalNoiseModel:
        return PhysicalNoiseModel(
            t_1=100,
            t_2=100,
            time_1_qubit_gate=0,
            time_2_qubit_gate=0,
            time_measurement=0,
            time_reset=0,
            p_1_qubit_gate_error=self._get_value("Gate error"),
            p_2_qubit_gate_error=self._get_value("Gate error"),
            p_reset_error=self._get_value("Collapsing operation error"),
            p_meas_qubit_error=self._get_value("Collapsing operation error"),
            p_readout_flip=self._get_value("Collapsing operation error"),
        )

    @override
    def apply(self, computation: Circuit) -> Circuit:
        noise_parameters, native_gates_times = (
            physical_noise_model_to_noise_parameters_and_native_gate_set_and_times(
                self._to_physical_noise_model()
            )
        )
        qpu = QPU(computation.qubits, native_gates_times, noise_parameters)
        return qpu.compile_and_add_noise_to_circuit(computation)

    @staticmethod
    def _check_is_probability(v: float, name: str) -> None:
        if v < 0 or v > 1:
            raise ValueError(f"Expected 0 <= {name.capitalize()} <= 1 but got {v}.")

    @override
    @classmethod
    def is_valid(cls, parameters: npt.NDArray[numpy.float64]) -> str | None:
        for i in range(2):
            SimplerNoise._check_is_probability(
                parameters[i], SimplerNoise.parameter_names[i]
            )
        return super().is_valid(parameters)
