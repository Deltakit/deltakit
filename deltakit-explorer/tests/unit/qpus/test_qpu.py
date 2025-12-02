# (c) Copyright Riverlane 2020-2025.
import pytest
from deltakit_circuit import (Circuit, Detector, GateLayer, MeasurementRecord,
                              NoiseLayer, Observable, Qubit, ShiftCoordinates,
                              measurement_noise_profile)
from deltakit_circuit.gates import (CX, CZ, MX, MZ, RX, RZ, H, I,
                                    OneQubitMeasurementGate, OneQubitResetGate,
                                    X)
from deltakit_circuit.noise_channels import Depolarise1, Depolarise2
from deltakit_explorer.qpu import QPU
from deltakit_explorer.qpu._native_gate_set import (NativeGateSet,
                                                    NativeGateSetAndTimes)
from deltakit_explorer.qpu._noise import (NoiseParameters,
                                          PhenomenologicalNoise, ToyNoise)

common_noise_model = ToyNoise(p=0.001)


@pytest.fixture(scope="module", params=[True, False])
def remove_paulis(request) -> bool:
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def maximise_parallelism(request) -> bool:
    return request.param


@pytest.fixture
def qpu_with_times(request):
    depolarise1_generator = Depolarise1.generator_from_prob(0.001)
    depolarise2_generator = Depolarise2.generator_from_prob(0.01)

    if hasattr(request, "param"):
        num_qubits = request.param.get("num_qubits", 6)
        maximise_parallelism = request.param["maximise_parallelism"]
    else:
        num_qubits = 6
        maximise_parallelism = True

    return QPU(
        qubits=[Qubit(i) for i in range(num_qubits)],
        native_gates_and_times=NativeGateSetAndTimes(
            one_qubit_gates={H: 40e-9},
            two_qubit_gates={CZ: 100e-9},
            reset_gates={RZ: 200e-9},
            measurement_gates={MZ: 150e-9},
        ),
        noise_model=NoiseParameters(
            gate_noise=[
                lambda noise_context: depolarise1_generator(
                    noise_context.gate_layer_qubits(H)
                ),
                lambda noise_context: depolarise2_generator(
                    noise_context.gate_layer_qubits(CZ)
                ),
            ],
            idle_noise=lambda qubit, t: Depolarise1(qubit, t * 1e5),
            reset_noise=[
                lambda noise_context: depolarise1_generator(
                    noise_context.gate_layer_qubits(OneQubitResetGate)
                )
            ],
            measurement_noise_after=[
                lambda noise_context: depolarise1_generator(
                    noise_context.gate_layer_qubits(OneQubitMeasurementGate)
                )
            ],
            measurement_flip=measurement_noise_profile(0.001),
        ),
        maximise_parallelism=maximise_parallelism,
    )


@pytest.fixture
def qpu_with_phenom_noise():
    return QPU(
        qubits=[Qubit(i) for i in range(10)],
        native_gates_and_times=NativeGateSet(),
        noise_model=PhenomenologicalNoise(
            phenomenological_noise=lambda qubit: Depolarise1(qubit, 0.001),
        ),
    )


@pytest.fixture
def circuit_for_compilation():
    return Circuit(
        [
            GateLayer([RZ(i) for i in range(10)]),
            GateLayer([H(Qubit(2 * i + 1)) for i in range(5)]),
            GateLayer([CZ(Qubit(2 * i), Qubit(2 * i + 1)) for i in range(5)]),
            GateLayer([MZ(i) for i in range(10)]),
        ]
    )


@pytest.fixture
def circuit_for_compilation_with_repeat_block():
    return Circuit(
        [
            GateLayer([RZ(i) for i in range(9)]),
            GateLayer([H(Qubit(2 * i + 1)) for i in range(4)]),
            GateLayer([CZ(Qubit(2 * i + 1), Qubit(2 * i)) for i in range(4)]),
            GateLayer([CZ(Qubit(2 * i + 1), Qubit(2 * i + 2)) for i in range(4)]),
            GateLayer([H(Qubit(2 * i + 1)) for i in range(4)]),
            GateLayer([MZ(2 * i + 1) for i in range(4)]),
            Detector([MeasurementRecord(-4)]),
            Detector([MeasurementRecord(-3)]),
            Detector([MeasurementRecord(-2)]),
            Detector([MeasurementRecord(-1)]),
            Circuit(
                [
                    GateLayer([RZ(2 * i + 1) for i in range(4)]),
                    GateLayer([H(Qubit(2 * i + 1)) for i in range(4)]),
                    GateLayer([CZ(Qubit(2 * i + 1), Qubit(2 * i)) for i in range(4)]),
                    GateLayer(
                        [CZ(Qubit(2 * i + 1), Qubit(2 * i + 2)) for i in range(4)]
                    ),
                    GateLayer([H(Qubit(2 * i + 1)) for i in range(4)]),
                    GateLayer([MZ(2 * i + 1) for i in range(4)]),
                    Detector([MeasurementRecord(-4), MeasurementRecord(-8)]),
                    Detector([MeasurementRecord(-3), MeasurementRecord(-7)]),
                    Detector([MeasurementRecord(-2), MeasurementRecord(-6)]),
                    Detector([MeasurementRecord(-1), MeasurementRecord(-5)]),
                ],
                iterations=3,
            ),
            GateLayer([RZ(2 * i + 1) for i in range(4)]),
            GateLayer([H(Qubit(2 * i + 1)) for i in range(4)]),
            GateLayer([CZ(Qubit(2 * i + 1), Qubit(2 * i)) for i in range(4)]),
            GateLayer([CZ(Qubit(2 * i + 1), Qubit(2 * i + 2)) for i in range(4)]),
            GateLayer([H(Qubit(2 * i + 1)) for i in range(4)]),
            GateLayer([MZ(i) for i in range(9)]),
            Detector([MeasurementRecord(-8), MeasurementRecord(-13)]),
            Detector([MeasurementRecord(-6), MeasurementRecord(-12)]),
            Detector([MeasurementRecord(-4), MeasurementRecord(-11)]),
            Detector([MeasurementRecord(-2), MeasurementRecord(-10)]),
            Detector(
                [MeasurementRecord(-8), MeasurementRecord(-9), MeasurementRecord(-7)]
            ),
            Detector(
                [MeasurementRecord(-6), MeasurementRecord(-7), MeasurementRecord(-5)]
            ),
            Detector(
                [MeasurementRecord(-4), MeasurementRecord(-5), MeasurementRecord(-3)]
            ),
            Detector(
                [MeasurementRecord(-2), MeasurementRecord(-3), MeasurementRecord(-1)]
            ),
            Observable(0, [MeasurementRecord(-9)]),
        ]
    )


@pytest.fixture
def circuit_for_compilation_with_paulis():
    return Circuit(
        [
            GateLayer([RZ(i) for i in range(10)]),
            GateLayer([CX(Qubit(2 * i), Qubit(2 * i + 1)) for i in range(5)]),
            GateLayer([H(Qubit(2 * i + 1)) for i in range(5)]),
            GateLayer([X(Qubit(2 * i + 1)) for i in range(5)]),
            GateLayer([MZ(i) for i in range(10)]),
        ]
    )


@pytest.fixture
def circuit_for_compilation_with_identities():
    return Circuit(
        [
            GateLayer([RZ(i) for i in range(10)]),
            GateLayer([I(i) for i in range(8)]),
            GateLayer([H(Qubit(2 * i + 1)) for i in range(5)]),
            GateLayer([CZ(Qubit(2 * i), Qubit(2 * i + 1)) for i in range(5)]),
            GateLayer([MZ(i) for i in range(10)]),
        ]
    )


@pytest.fixture
def circuit_for_compilation_without_H():
    return Circuit(
        [
            GateLayer([RX(i) for i in range(10)]),
            GateLayer([CX(Qubit(2 * i), Qubit(2 * i + 1)) for i in range(5)]),
            GateLayer([I(Qubit(i)) for i in range(8)]),
            GateLayer([MZ(i) for i in range(10)]),
        ]
    )


@pytest.fixture
def original_and_noise_compiled_circuits_with_qpu_time(request):
    return [
        (
            Circuit(
                [
                    GateLayer([RZ(0), RZ(1)]),
                    GateLayer([MZ(0), MZ(1)]),
                    GateLayer([RZ(0), RZ(1)]),
                    Circuit(
                        [
                            GateLayer([RX(2), RX(3)]),
                            GateLayer([CZ(2, 0), CZ(3, 1)]),
                            GateLayer([MX(2), MX(3)]),
                            Detector(
                                [MeasurementRecord(-2), MeasurementRecord(-4)],
                                (0, 1),
                            ),
                            Detector(
                                [MeasurementRecord(-1), MeasurementRecord(-3)],
                                (1, 1),
                            ),
                            ShiftCoordinates((0, 1)),
                        ],
                        iterations=3,
                    ),
                    GateLayer([MZ(0), MZ(1)]),
                ]
            ),
            Circuit(
                [
                    GateLayer([RZ(0), RZ(1)]),
                    NoiseLayer([Depolarise1(0, 0.001), Depolarise1(1, 0.001)]),
                    NoiseLayer(
                        [
                            Depolarise1(2, 0.02),
                            Depolarise1(3, 0.02),
                            Depolarise1(4, 0.02),
                            Depolarise1(5, 0.02),
                        ]
                    ),
                    GateLayer([MZ(0, 0.001), MZ(1, 0.001)]),
                    NoiseLayer([Depolarise1(0, 0.001), Depolarise1(1, 0.001)]),
                    NoiseLayer(
                        [
                            Depolarise1(2, 0.015),
                            Depolarise1(3, 0.015),
                            Depolarise1(4, 0.015),
                            Depolarise1(5, 0.015),
                        ]
                    ),
                    GateLayer([RZ(0), RZ(1)]),
                    NoiseLayer([Depolarise1(0, 0.001), Depolarise1(1, 0.001)]),
                    NoiseLayer(
                        [
                            Depolarise1(2, 0.02),
                            Depolarise1(3, 0.02),
                            Depolarise1(4, 0.02),
                            Depolarise1(5, 0.02),
                        ]
                    ),
                    Circuit(
                        [
                            GateLayer([RZ(2), RZ(3)]),
                            NoiseLayer([Depolarise1(2, 0.001), Depolarise1(3, 0.001)]),
                            NoiseLayer(
                                [
                                    Depolarise1(0, 0.02),
                                    Depolarise1(1, 0.02),
                                    Depolarise1(4, 0.02),
                                    Depolarise1(5, 0.02),
                                ]
                            ),
                            GateLayer([H(2), H(3)]),
                            NoiseLayer([Depolarise1(2, 0.001), Depolarise1(3, 0.001)]),
                            NoiseLayer(
                                [
                                    Depolarise1(0, 0.004),
                                    Depolarise1(1, 0.004),
                                    Depolarise1(4, 0.004),
                                    Depolarise1(5, 0.004),
                                ]
                            ),
                            GateLayer([CZ(2, 0), CZ(3, 1)]),
                            NoiseLayer(
                                [
                                    Depolarise2(2, 0, 0.01),
                                    Depolarise2(3, 1, 0.01),
                                ]
                            ),
                            NoiseLayer([Depolarise1(4, 0.01), Depolarise1(5, 0.01)]),
                            GateLayer([H(2), H(3)]),
                            NoiseLayer([Depolarise1(2, 0.001), Depolarise1(3, 0.001)]),
                            NoiseLayer(
                                [
                                    Depolarise1(0, 0.004),
                                    Depolarise1(1, 0.004),
                                    Depolarise1(4, 0.004),
                                    Depolarise1(5, 0.004),
                                ]
                            ),
                            GateLayer([MZ(2, 0.001), MZ(3, 0.001)]),
                            NoiseLayer([Depolarise1(2, 0.001), Depolarise1(3, 0.001)]),
                            Detector(
                                [MeasurementRecord(-2), MeasurementRecord(-4)],
                                (0, 1),
                            ),
                            Detector(
                                [MeasurementRecord(-1), MeasurementRecord(-3)],
                                (1, 1),
                            ),
                            ShiftCoordinates((0, 1)),
                            NoiseLayer(
                                [
                                    Depolarise1(0, 0.015),
                                    Depolarise1(1, 0.015),
                                    Depolarise1(4, 0.015),
                                    Depolarise1(5, 0.015),
                                ]
                            ),
                            GateLayer([H(2), H(3)]),
                            NoiseLayer([Depolarise1(2, 0.001), Depolarise1(3, 0.001)]),
                            NoiseLayer(
                                [
                                    Depolarise1(0, 0.004),
                                    Depolarise1(1, 0.004),
                                    Depolarise1(4, 0.004),
                                    Depolarise1(5, 0.004),
                                ]
                            ),
                        ],
                        iterations=3,
                    ),
                    GateLayer([MZ(0, 0.001), MZ(1, 0.001)]),
                    NoiseLayer([Depolarise1(0, 0.001), Depolarise1(1, 0.001)]),
                    NoiseLayer(
                        [
                            Depolarise1(2, 0.015),
                            Depolarise1(3, 0.015),
                            Depolarise1(4, 0.015),
                            Depolarise1(5, 0.015),
                        ]
                    ),
                ]
            ),
            2410.0e-9,
            # RZ+MZ+RZ+ (RZ+H+CZ+H+MZ+H)*3 + MZ =
            # 200+150+200 + (200+40+100+40+150+40)*3 + 150 =
            # 550+ 570*3 + 150 = 2410
            Circuit(
                [
                    GateLayer([RZ(0), RZ(1)]),
                    NoiseLayer([Depolarise1(0, 0.001), Depolarise1(1, 0.001)]),
                    NoiseLayer(
                        [
                            Depolarise1(2, 0.02),
                            Depolarise1(3, 0.02),
                            Depolarise1(4, 0.02),
                            Depolarise1(5, 0.02),
                        ]
                    ),
                    GateLayer([MZ(0, 0.001), MZ(1, 0.001)]),
                    NoiseLayer([Depolarise1(0, 0.001), Depolarise1(1, 0.001)]),
                    NoiseLayer(
                        [
                            Depolarise1(2, 0.015),
                            Depolarise1(3, 0.015),
                            Depolarise1(4, 0.015),
                            Depolarise1(5, 0.015),
                        ]
                    ),
                    GateLayer([RZ(0), RZ(1)]),
                    NoiseLayer([Depolarise1(0, 0.001), Depolarise1(1, 0.001)]),
                    NoiseLayer(
                        [
                            Depolarise1(2, 0.02),
                            Depolarise1(3, 0.02),
                            Depolarise1(4, 0.02),
                            Depolarise1(5, 0.02),
                        ]
                    ),
                    Circuit(
                        [
                            GateLayer([RZ(2), RZ(3)]),
                            NoiseLayer([Depolarise1(2, 0.001), Depolarise1(3, 0.001)]),
                            NoiseLayer(
                                [
                                    Depolarise1(0, 0.02),
                                    Depolarise1(1, 0.02),
                                    Depolarise1(4, 0.02),
                                    Depolarise1(5, 0.02),
                                ]
                            ),
                            GateLayer([H(2), H(3)]),
                            NoiseLayer([Depolarise1(2, 0.001), Depolarise1(3, 0.001)]),
                            NoiseLayer(
                                [
                                    Depolarise1(0, 0.004),
                                    Depolarise1(1, 0.004),
                                    Depolarise1(4, 0.004),
                                    Depolarise1(5, 0.004),
                                ]
                            ),
                            GateLayer([CZ(2, 0), CZ(3, 1)]),
                            NoiseLayer(
                                [
                                    Depolarise2(2, 0, 0.01),
                                    Depolarise2(3, 1, 0.01),
                                ]
                            ),
                            NoiseLayer([Depolarise1(4, 0.01), Depolarise1(5, 0.01)]),
                            GateLayer([H(2), H(3)]),
                            NoiseLayer([Depolarise1(2, 0.001), Depolarise1(3, 0.001)]),
                            NoiseLayer(
                                [
                                    Depolarise1(0, 0.004),
                                    Depolarise1(1, 0.004),
                                    Depolarise1(4, 0.004),
                                    Depolarise1(5, 0.004),
                                ]
                            ),
                            GateLayer([MZ(2, 0.001), MZ(3, 0.001)]),
                            NoiseLayer([Depolarise1(2, 0.001), Depolarise1(3, 0.001)]),
                            Detector(
                                [MeasurementRecord(-2), MeasurementRecord(-4)],
                                (0, 1),
                            ),
                            Detector(
                                [MeasurementRecord(-1), MeasurementRecord(-3)],
                                (1, 1),
                            ),
                            ShiftCoordinates((0, 1)),
                            NoiseLayer(
                                [
                                    Depolarise1(0, 0.015),
                                    Depolarise1(1, 0.015),
                                    Depolarise1(4, 0.015),
                                    Depolarise1(5, 0.015),
                                ]
                            ),
                            GateLayer([H(2), H(3)]),
                            NoiseLayer([Depolarise1(2, 0.001), Depolarise1(3, 0.001)]),
                            NoiseLayer(
                                [
                                    Depolarise1(0, 0.004),
                                    Depolarise1(1, 0.004),
                                    Depolarise1(4, 0.004),
                                    Depolarise1(5, 0.004),
                                ]
                            ),
                        ],
                        iterations=3,
                    ),
                    GateLayer([MZ(0, 0.001), MZ(1, 0.001)]),
                    NoiseLayer([Depolarise1(0, 0.001), Depolarise1(1, 0.001)]),
                    NoiseLayer(
                        [
                            Depolarise1(2, 0.015),
                            Depolarise1(3, 0.015),
                            Depolarise1(4, 0.015),
                            Depolarise1(5, 0.015),
                        ]
                    ),
                ]
            ),
            2410.0e-9,
            # RZ+MZ+RZ+ (RZ+H+CZ+H+MZ+H)*3 + MZ =
            # 200+150+200 + (200+40+100+40+150+40)*3 + 150 =
            # 550+ 570*3 + 150 = 2410
        ),
        (
            Circuit(
                [
                    GateLayer([RX(i) for i in range(6)]),
                    GateLayer([CX(4, 0), CZ(5, 2)]),
                    GateLayer([CX(4, 1), CZ(5, 3)]),
                    GateLayer(CX(4, 2)),
                    GateLayer(MX(5)),
                    GateLayer(CX(4, 3)),
                    GateLayer(MX(4)),
                    Detector([MeasurementRecord(-1)], (0, 0)),
                ]
            ),
            Circuit(
                [
                    GateLayer([RZ(i) for i in range(6)]),
                    NoiseLayer([Depolarise1(i, 0.001) for i in range(6)]),
                    GateLayer([H(i) for i in range(2, 6)]),
                    NoiseLayer([Depolarise1(i, 0.001) for i in range(2, 6)]),
                    NoiseLayer([Depolarise1(0, 0.004), Depolarise1(1, 0.004)]),
                    GateLayer([CZ(4, 0), CZ(5, 2)]),
                    NoiseLayer([Depolarise2(4, 0, 0.01), Depolarise2(5, 2, 0.01)]),
                    NoiseLayer([Depolarise1(1, 0.01), Depolarise1(3, 0.01)]),
                    GateLayer([CZ(4, 1), CZ(5, 3), H(0), H(2)]),
                    NoiseLayer(
                        [
                            Depolarise2(4, 1, 0.01),
                            Depolarise2(5, 3, 0.01),
                            Depolarise1(0, 0.001),
                            Depolarise1(2, 0.001),
                        ]
                    ),
                    NoiseLayer([Depolarise1(0, 0.006), Depolarise1(2, 0.006)]),
                    GateLayer([CZ(4, 2), H(1), H(3), H(5)]),
                    NoiseLayer(
                        [
                            Depolarise2(4, 2, 0.01),
                            Depolarise1(1, 0.001),
                            Depolarise1(3, 0.001),
                            Depolarise1(5, 0.001),
                        ]
                    ),
                    NoiseLayer([Depolarise1(0, 0.004)]),
                    GateLayer(MZ(5, 0.001)),
                    NoiseLayer([Depolarise1(5, 0.001)]),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.006),
                            Depolarise1(1, 0.006),
                            Depolarise1(3, 0.006),
                        ]
                    ),
                    GateLayer(CZ(4, 3)),
                    NoiseLayer([Depolarise2(4, 3, 0.01)]),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.01),
                            Depolarise1(1, 0.01),
                            Depolarise1(2, 0.01),
                            Depolarise1(5, 0.001),
                        ]
                    ),
                    GateLayer([H(2), H(3), H(4), H(5)]),
                    NoiseLayer(
                        [
                            Depolarise1(2, 0.001),
                            Depolarise1(3, 0.001),
                            Depolarise1(4, 0.001),
                            Depolarise1(5, 0.001),
                        ]
                    ),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.004),
                            Depolarise1(1, 0.004),
                        ]
                    ),
                    GateLayer(MZ(4, 0.001)),
                    NoiseLayer(Depolarise1(4, 0.001)),
                    Detector([MeasurementRecord(-1)], (0, 0)),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.015),
                            Depolarise1(1, 0.015),
                            Depolarise1(2, 0.015),
                            Depolarise1(3, 0.015),
                            Depolarise1(5, 0.015),
                        ]
                    ),
                    GateLayer(H(4)),
                    NoiseLayer(
                        [
                            Depolarise1(4, 0.001),
                        ]
                    ),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.004),
                            Depolarise1(1, 0.004),
                            Depolarise1(2, 0.004),
                            Depolarise1(3, 0.004),
                            Depolarise1(5, 0.004),
                        ]
                    ),
                ]
            ),
            870e-9,
            # RZ+H+CZ+CZ+CZ+MIN(MZ,CZ)+H+MZ+H =
            # ** MZ is parallel to the rest of the circuit
            # 200+40+100*3+100+40+150+40 =
            # 870
            Circuit(
                [
                    GateLayer([RZ(i) for i in range(6)]),
                    NoiseLayer([Depolarise1(i, 0.001) for i in range(6)]),
                    GateLayer([H(i) for i in range(2, 6)]),
                    NoiseLayer([Depolarise1(i, 0.001) for i in range(2, 6)]),
                    NoiseLayer([Depolarise1(0, 0.004), Depolarise1(1, 0.004)]),
                    GateLayer([CZ(4, 0), CZ(5, 2)]),
                    NoiseLayer([Depolarise2(4, 0, 0.01), Depolarise2(5, 2, 0.01)]),
                    NoiseLayer([Depolarise1(1, 0.01), Depolarise1(3, 0.01)]),
                    GateLayer([CZ(4, 1), CZ(5, 3), H(0), H(2)]),
                    NoiseLayer(
                        [
                            Depolarise2(4, 1, 0.01),
                            Depolarise2(5, 3, 0.01),
                            Depolarise1(0, 0.001),
                            Depolarise1(2, 0.001),
                        ]
                    ),
                    NoiseLayer([Depolarise1(0, 0.006), Depolarise1(2, 0.006)]),
                    GateLayer([CZ(4, 2), H(1), H(3), H(5)]),
                    NoiseLayer(
                        [
                            Depolarise2(4, 2, 0.01),
                            Depolarise1(1, 0.001),
                            Depolarise1(3, 0.001),
                            Depolarise1(5, 0.001),
                        ]
                    ),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.01),
                            Depolarise1(1, 0.006),
                            Depolarise1(3, 0.006),
                            Depolarise1(5, 0.006),
                        ]
                    ),
                    GateLayer(MZ(5, 0.001)),
                    NoiseLayer([Depolarise1(5, 0.001)]),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.015),
                            Depolarise1(1, 0.015),
                            Depolarise1(2, 0.015),
                            Depolarise1(3, 0.015),
                            Depolarise1(4, 0.015),
                        ]
                    ),
                    GateLayer(CZ(4, 3)),
                    NoiseLayer([Depolarise2(4, 3, 0.01)]),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.01),
                            Depolarise1(1, 0.01),
                            Depolarise1(2, 0.01),
                            Depolarise1(5, 0.01),
                        ]
                    ),
                    GateLayer([H(2), H(3), H(4), H(5)]),
                    NoiseLayer(
                        [
                            Depolarise1(2, 0.001),
                            Depolarise1(3, 0.001),
                            Depolarise1(4, 0.001),
                            Depolarise1(5, 0.001),
                        ]
                    ),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.004),
                            Depolarise1(1, 0.004),
                        ]
                    ),
                    GateLayer(MZ(4, 0.001)),
                    NoiseLayer(Depolarise1(4, 0.001)),
                    Detector([MeasurementRecord(-1)], (0, 0)),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.015),
                            Depolarise1(1, 0.015),
                            Depolarise1(2, 0.015),
                            Depolarise1(3, 0.015),
                            Depolarise1(5, 0.015),
                        ]
                    ),
                    GateLayer(H(4)),
                    NoiseLayer(
                        [
                            Depolarise1(4, 0.001),
                        ]
                    ),
                    NoiseLayer(
                        [
                            Depolarise1(0, 0.004),
                            Depolarise1(1, 0.004),
                            Depolarise1(2, 0.004),
                            Depolarise1(3, 0.004),
                            Depolarise1(5, 0.004),
                        ]
                    ),
                ]
            ),
            1020e-9,
            # RZ+H+CZ+CZ+CZ+MZ+CZ+H+MZ+H =
            # 200+40+100*3+150+100+40+150+40 =
            # 1020
        ),
    ][request.param]
