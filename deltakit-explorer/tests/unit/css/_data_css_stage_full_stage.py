from dataclasses import dataclass

from deltakit_circuit import (Circuit, Detector, GateLayer, MeasurementRecord,
                              Observable, PauliX, PauliZ, Qubit,
                              ShiftCoordinates)
from deltakit_circuit._basic_types import Coord2D
from deltakit_circuit.gates import CX, CZ, MPP, MX, MZ, RX, RZ, I
from deltakit_explorer.codes._css._css_stage import CSSStage
from deltakit_explorer.codes._stabiliser import Stabiliser

from ._data_css_stage import (example_simultaneous_stabilisers,
                              example_spaced_stabilisers)

# creating a default value different from None to avoid tests expecting None to accidentally
# pass on test components which haven't been explicitly defined.
MISSING_VALUE = "MISSING_VALUE"


@dataclass
class CSSStageTestComponents:
    stage: CSSStage
    expected_first_round: Circuit = MISSING_VALUE  # type: ignore [assignment]
    expected_remaining_rounds: Circuit = MISSING_VALUE  # type: ignore [assignment]
    expected_measurements_as_stabilisers: tuple[Stabiliser, ...] = MISSING_VALUE  # type: ignore [assignment]
    expected_resets_as_stabilisers: tuple[Stabiliser, ...] = MISSING_VALUE  # type: ignore [assignment]
    expected_ordered_stabilisers: tuple[Stabiliser, ...] = MISSING_VALUE  # type: ignore [assignment]


full_stage_1_round = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=1,
        stabilisers=example_simultaneous_stabilisers,
        first_round_measurements=[
            MX(Qubit(Coord2D(0, 0))),
            MZ(Qubit(Coord2D(0, 4))),
            MX(Qubit(Coord2D(0, 6))),
        ],
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
        observable_definitions={0: [Qubit(Coord2D(0, 0))]},
    ),
    expected_first_round=Circuit(
        [
            GateLayer(
                [
                    MX(Qubit(Coord2D(0, 0))),
                    MZ(Qubit(Coord2D(0, 4))),
                    MX(Qubit(Coord2D(0, 6))),
                ],
            ),
            GateLayer(
                {
                    RX(Qubit(Coord2D(2, 0))),
                    RX(Qubit(Coord2D(2, 2))),
                    RX(Qubit(Coord2D(2, 4))),
                }
            ),
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 1))),
                    I(Qubit(Coord2D(1, 3))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(3, 3))),
                    CZ(Qubit(Coord2D(2, 0)), Qubit(Coord2D(3, 1))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(3, 1))),
                    CZ(Qubit(Coord2D(2, 0)), Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(1, 3))),
                    CZ(Qubit(Coord2D(2, 4)), Qubit(Coord2D(3, 3))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(1, 1))),
                    CZ(Qubit(Coord2D(2, 4)), Qubit(Coord2D(1, 3))),
                }
            ),
            GateLayer(
                [
                    MX(Qubit(Coord2D(2, 2))),
                    MX(Qubit(Coord2D(2, 4))),
                    MX(Qubit(Coord2D(2, 0))),
                ]
            ),
            Observable(0, [MeasurementRecord(-6)]),
        ]
    ),
    expected_remaining_rounds=Circuit(
        GateLayer(
            {
                RX(Qubit(Coord2D(0, 0))),
                RZ(Qubit(Coord2D(0, 4))),
                RX(Qubit(Coord2D(0, 6))),
            }
        )
    ),
    expected_measurements_as_stabilisers=(
        Stabiliser(
            paulis=(PauliX(Qubit(Coord2D(0, 0))),),
            ancilla_qubit=Qubit(Coord2D(0, 0)),
        ),
        Stabiliser(
            paulis=(PauliZ(Qubit(Coord2D(0, 4))),),
            ancilla_qubit=Qubit(Coord2D(0, 4)),
        ),
        Stabiliser(
            paulis=(PauliX(Qubit(Coord2D(0, 6))),),
            ancilla_qubit=Qubit(Coord2D(0, 6)),
        ),
    ),
    expected_resets_as_stabilisers=(
        Stabiliser(
            paulis=(PauliX(Qubit(Coord2D(0, 0))),),
            ancilla_qubit=Qubit(Coord2D(0, 0)),
        ),
        Stabiliser(
            paulis=(PauliZ(Qubit(Coord2D(0, 4))),),
            ancilla_qubit=Qubit(Coord2D(0, 4)),
        ),
        Stabiliser(
            paulis=(PauliX(Qubit(Coord2D(0, 6))),),
            ancilla_qubit=Qubit(Coord2D(0, 6)),
        ),
    ),
    expected_ordered_stabilisers=(
        Stabiliser(
            paulis=(
                PauliX(Coord2D(3, 3)),
                PauliX(Coord2D(3, 1)),
                PauliX(Coord2D(1, 3)),
                PauliX(Coord2D(1, 1)),
            ),
            ancilla_qubit=Qubit(Coord2D(2, 2)),
        ),
        Stabiliser(
            paulis=(
                None,
                None,
                PauliZ(Coord2D(3, 3)),
                PauliZ(Coord2D(1, 3)),
            ),
            ancilla_qubit=Qubit(Coord2D(2, 4)),
        ),
        Stabiliser(
            paulis=(
                PauliZ(Coord2D(3, 1)),
                PauliZ(Coord2D(1, 1)),
                None,
                None,
            ),
            ancilla_qubit=Qubit(Coord2D(2, 0)),
        ),
    ),
)

full_stage_1_round_with_mpps = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=1,
        stabilisers=example_simultaneous_stabilisers,
        first_round_measurements=[
            MPP([PauliX(Coord2D(1, 1)), PauliX(Coord2D(3, 1))]),
            MX(Qubit(Coord2D(0, 0))),
            MZ(Qubit(Coord2D(0, 4))),
            MPP([PauliX(Coord2D(1, 3)), PauliX(Coord2D(3, 3))]),
            MX(Qubit(Coord2D(0, 6))),
        ],
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
        observable_definitions={
            0: [
                Qubit(Coord2D(0, 0)),
                MPP([PauliX(Coord2D(1, 1)), PauliX(Coord2D(3, 1))]),
            ]
        },
    ),
    expected_first_round=Circuit(
        [
            GateLayer(
                [
                    MPP([PauliX(Coord2D(1, 1)), PauliX(Coord2D(3, 1))]),
                    MX(Qubit(Coord2D(0, 0))),
                    MZ(Qubit(Coord2D(0, 4))),
                    MPP([PauliX(Coord2D(1, 3)), PauliX(Coord2D(3, 3))]),
                    MX(Qubit(Coord2D(0, 6))),
                ],
            ),
            GateLayer(
                {
                    RX(Qubit(Coord2D(2, 0))),
                    RX(Qubit(Coord2D(2, 2))),
                    RX(Qubit(Coord2D(2, 4))),
                }
            ),
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 1))),
                    I(Qubit(Coord2D(1, 3))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(3, 3))),
                    CZ(Qubit(Coord2D(2, 0)), Qubit(Coord2D(3, 1))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(3, 1))),
                    CZ(Qubit(Coord2D(2, 0)), Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(1, 3))),
                    CZ(Qubit(Coord2D(2, 4)), Qubit(Coord2D(3, 3))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(1, 1))),
                    CZ(Qubit(Coord2D(2, 4)), Qubit(Coord2D(1, 3))),
                }
            ),
            GateLayer(
                [
                    MX(Qubit(Coord2D(2, 2))),
                    MX(Qubit(Coord2D(2, 4))),
                    MX(Qubit(Coord2D(2, 0))),
                ]
            ),
            Observable(0, [MeasurementRecord(-7), MeasurementRecord(-8)]),
        ]
    ),
    expected_remaining_rounds=Circuit(
        GateLayer(
            {
                RX(Qubit(Coord2D(0, 0))),
                RZ(Qubit(Coord2D(0, 4))),
                RX(Qubit(Coord2D(0, 6))),
            }
        )
    ),
    expected_measurements_as_stabilisers=(
        Stabiliser([PauliX(Coord2D(1, 1)), PauliX(Coord2D(3, 1))]),
        Stabiliser([PauliX(Coord2D(1, 3)), PauliX(Coord2D(3, 3))]),
        Stabiliser(
            paulis=(PauliX(Qubit(Coord2D(0, 0))),),
            ancilla_qubit=Qubit(Coord2D(0, 0)),
        ),
        Stabiliser(
            paulis=(PauliZ(Qubit(Coord2D(0, 4))),),
            ancilla_qubit=Qubit(Coord2D(0, 4)),
        ),
        Stabiliser(
            paulis=(PauliX(Qubit(Coord2D(0, 6))),),
            ancilla_qubit=Qubit(Coord2D(0, 6)),
        ),
    ),
    expected_resets_as_stabilisers=(
        Stabiliser(
            paulis=(PauliX(Qubit(Coord2D(0, 0))),),
            ancilla_qubit=Qubit(Coord2D(0, 0)),
        ),
        Stabiliser(
            paulis=(PauliZ(Qubit(Coord2D(0, 4))),),
            ancilla_qubit=Qubit(Coord2D(0, 4)),
        ),
        Stabiliser(
            paulis=(PauliX(Qubit(Coord2D(0, 6))),),
            ancilla_qubit=Qubit(Coord2D(0, 6)),
        ),
    ),
    expected_ordered_stabilisers=(
        Stabiliser(
            paulis=(
                PauliX(Coord2D(3, 3)),
                PauliX(Coord2D(3, 1)),
                PauliX(Coord2D(1, 3)),
                PauliX(Coord2D(1, 1)),
            ),
            ancilla_qubit=Qubit(Coord2D(2, 2)),
        ),
        Stabiliser(
            paulis=(
                None,
                None,
                PauliZ(Coord2D(3, 3)),
                PauliZ(Coord2D(1, 3)),
            ),
            ancilla_qubit=Qubit(Coord2D(2, 4)),
        ),
        Stabiliser(
            paulis=(
                PauliZ(Coord2D(3, 1)),
                PauliZ(Coord2D(1, 1)),
                None,
                None,
            ),
            ancilla_qubit=Qubit(Coord2D(2, 0)),
        ),
    ),
)


full_stage_4_rounds = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=4,
        stabilisers=example_simultaneous_stabilisers,
        first_round_measurements=[
            MX(Qubit(Coord2D(0, 0))),
            MZ(Qubit(Coord2D(0, 4))),
            MX(Qubit(Coord2D(0, 6))),
        ],
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
        observable_definitions={0: [Qubit(Coord2D(0, 0))]},
    ),
    expected_first_round=Circuit(
        [
            GateLayer(
                [
                    MX(Qubit(Coord2D(0, 0))),
                    MZ(Qubit(Coord2D(0, 4))),
                    MX(Qubit(Coord2D(0, 6))),
                ],
            ),
            GateLayer(
                {
                    RX(Qubit(Coord2D(2, 0))),
                    RX(Qubit(Coord2D(2, 2))),
                    RX(Qubit(Coord2D(2, 4))),
                }
            ),
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 1))),
                    I(Qubit(Coord2D(1, 3))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(3, 3))),
                    CZ(Qubit(Coord2D(2, 0)), Qubit(Coord2D(3, 1))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(3, 1))),
                    CZ(Qubit(Coord2D(2, 0)), Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(1, 3))),
                    CZ(Qubit(Coord2D(2, 4)), Qubit(Coord2D(3, 3))),
                }
            ),
            GateLayer(
                {
                    CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(1, 1))),
                    CZ(Qubit(Coord2D(2, 4)), Qubit(Coord2D(1, 3))),
                }
            ),
            GateLayer(
                [
                    MX(Qubit(Coord2D(2, 2))),
                    MX(Qubit(Coord2D(2, 4))),
                    MX(Qubit(Coord2D(2, 0))),
                ]
            ),
            Observable(0, [MeasurementRecord(-6)]),
        ]
    ),
    expected_remaining_rounds=Circuit(
        [
            Circuit(
                [
                    GateLayer(
                        [
                            RX(Qubit(Coord2D(2, 2))),
                            RX(Qubit(Coord2D(2, 4))),
                            RX(Qubit(Coord2D(2, 0))),
                        ]
                    ),
                    GateLayer(
                        [
                            I(Qubit(Coord2D(3, 1))),
                            I(Qubit(Coord2D(1, 1))),
                            I(Qubit(Coord2D(3, 3))),
                            I(Qubit(Coord2D(1, 3))),
                        ]
                    ),
                    GateLayer(
                        [
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(3, 3)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 0)),
                                Qubit(Coord2D(3, 1)),
                            ),
                        ]
                    ),
                    GateLayer(
                        [
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(3, 1)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 0)),
                                Qubit(Coord2D(1, 1)),
                            ),
                        ]
                    ),
                    GateLayer(
                        [
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(1, 3)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 4)),
                                Qubit(Coord2D(3, 3)),
                            ),
                        ]
                    ),
                    GateLayer(
                        [
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(1, 1)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 4)),
                                Qubit(Coord2D(1, 3)),
                            ),
                        ]
                    ),
                    GateLayer(
                        [
                            MX(Qubit(Coord2D(2, 2))),
                            MX(Qubit(Coord2D(2, 4))),
                            MX(Qubit(Coord2D(2, 0))),
                        ]
                    ),
                    Detector([MeasurementRecord(-6), MeasurementRecord(-3)], (2, 2, 0)),
                    Detector([MeasurementRecord(-5), MeasurementRecord(-2)], (2, 4, 0)),
                    Detector([MeasurementRecord(-4), MeasurementRecord(-1)], (2, 0, 0)),
                    ShiftCoordinates((0, 0, 1)),
                ],
                iterations=3,
            ),
            GateLayer(
                [
                    RX(Qubit(Coord2D(0, 6))),
                    RX(Qubit(Coord2D(0, 0))),
                    RZ(Qubit(Coord2D(0, 4))),
                ]
            ),
        ],
    ),
)

full_stage_many_rounds_spaced = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=2,
        stabilisers=example_spaced_stabilisers,
        first_round_measurements=[
            MX(Qubit(Coord2D(0, 0))),
            MZ(Qubit(Coord2D(0, 4))),
            MX(Qubit(Coord2D(0, 6))),
        ],
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
        observable_definitions={0: [Qubit(Coord2D(0, 0))]},
    ),
    expected_first_round=Circuit(
        [
            GateLayer(
                [
                    MX(Qubit(Coord2D(0, 0))),
                    MZ(Qubit(Coord2D(0, 4))),
                    MX(Qubit(Coord2D(0, 6))),
                ],
            ),
            GateLayer(
                RX(Qubit(Coord2D(2, 2))),
            ),
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 1))),
                    I(Qubit(Coord2D(1, 3))),
                }
            ),
            GateLayer(
                CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(3, 3))),
            ),
            GateLayer(
                CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(3, 1))),
            ),
            GateLayer(
                CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(1, 3))),
            ),
            GateLayer(
                CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(1, 1))),
            ),
            GateLayer(
                MX(Qubit(Coord2D(2, 2))),
            ),
            GateLayer(
                {
                    RX(Qubit(Coord2D(2, 0))),
                    RX(Qubit(Coord2D(2, 4))),
                }
            ),
            GateLayer(
                CZ(Qubit(Coord2D(2, 0)), Qubit(Coord2D(3, 1))),
            ),
            GateLayer(
                CZ(Qubit(Coord2D(2, 0)), Qubit(Coord2D(1, 1))),
            ),
            GateLayer(
                CZ(Qubit(Coord2D(2, 4)), Qubit(Coord2D(3, 3))),
            ),
            GateLayer(
                CZ(Qubit(Coord2D(2, 4)), Qubit(Coord2D(1, 3))),
            ),
            GateLayer(
                [
                    MX(Qubit(Coord2D(2, 4))),
                    MX(Qubit(Coord2D(2, 0))),
                ]
            ),
            Observable(0, [MeasurementRecord(-6)]),
        ]
    ),
    expected_remaining_rounds=Circuit(
        [
            GateLayer(
                RX(Qubit(Coord2D(2, 2))),
            ),
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 1))),
                    I(Qubit(Coord2D(1, 3))),
                }
            ),
            GateLayer(
                CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(3, 3))),
            ),
            GateLayer(
                CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(3, 1))),
            ),
            GateLayer(
                CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(1, 3))),
            ),
            GateLayer(
                CX(Qubit(Coord2D(2, 2)), Qubit(Coord2D(1, 1))),
            ),
            GateLayer(
                MX(Qubit(Coord2D(2, 2))),
            ),
            GateLayer(
                {
                    RX(Qubit(Coord2D(2, 0))),
                    RX(Qubit(Coord2D(2, 4))),
                }
            ),
            GateLayer(
                CZ(Qubit(Coord2D(2, 0)), Qubit(Coord2D(3, 1))),
            ),
            GateLayer(
                CZ(Qubit(Coord2D(2, 0)), Qubit(Coord2D(1, 1))),
            ),
            GateLayer(
                CZ(Qubit(Coord2D(2, 4)), Qubit(Coord2D(3, 3))),
            ),
            GateLayer(
                CZ(Qubit(Coord2D(2, 4)), Qubit(Coord2D(1, 3))),
            ),
            GateLayer(
                [
                    MX(Qubit(Coord2D(2, 4))),
                    MX(Qubit(Coord2D(2, 0))),
                ]
            ),
            Detector({MeasurementRecord(-3), MeasurementRecord(-6)}, (2, 2, 0)),
            Detector({MeasurementRecord(-2), MeasurementRecord(-5)}, (2, 4, 0)),
            Detector({MeasurementRecord(-1), MeasurementRecord(-4)}, (2, 0, 0)),
            ShiftCoordinates((0, 0, 1)),
            GateLayer(
                {
                    RX(Qubit(Coord2D(0, 0))),
                    RZ(Qubit(Coord2D(0, 4))),
                    RX(Qubit(Coord2D(0, 6))),
                }
            ),
        ]
    ),
)

full_stage_many_rounds_no_ancilla = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=2,
        stabilisers=[
            [
                Stabiliser(stabiliser.paulis)
                for stabiliser in example_simultaneous_stabilisers[0]
            ]
        ],
        first_round_measurements=[
            MX(Qubit(Coord2D(0, 0))),
            MZ(Qubit(Coord2D(0, 4))),
            MX(Qubit(Coord2D(0, 6))),
        ],
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
        observable_definitions={0: [Qubit(Coord2D(0, 0))]},
    ),
    expected_first_round=Circuit(
        [
            GateLayer(
                [
                    MX(Qubit(Coord2D(0, 0))),
                    MZ(Qubit(Coord2D(0, 4))),
                    MX(Qubit(Coord2D(0, 6))),
                ],
            ),
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 3))),
                    I(Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                MPP(
                    (
                        PauliX(Coord2D(3, 3)),
                        PauliX(Coord2D(3, 1)),
                        PauliX(Coord2D(1, 3)),
                        PauliX(Coord2D(1, 1)),
                    )
                )
            ),
            GateLayer(
                [
                    MPP((PauliZ(Coord2D(3, 3)), PauliZ(Coord2D(1, 3)))),
                    MPP((PauliZ(Coord2D(3, 1)), PauliZ(Coord2D(1, 1)))),
                ]
            ),
            Observable(0, [MeasurementRecord(-6)]),
        ]
    ),
    expected_remaining_rounds=Circuit(
        [
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 3))),
                    I(Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                MPP(
                    (
                        PauliX(Coord2D(3, 3)),
                        PauliX(Coord2D(3, 1)),
                        PauliX(Coord2D(1, 3)),
                        PauliX(Coord2D(1, 1)),
                    )
                )
            ),
            GateLayer(
                [
                    MPP((PauliZ(Coord2D(3, 3)), PauliZ(Coord2D(1, 3)))),
                    MPP((PauliZ(Coord2D(3, 1)), PauliZ(Coord2D(1, 1)))),
                ]
            ),
            Detector({MeasurementRecord(-3), MeasurementRecord(-6)}, (2, 2, 0)),
            Detector({MeasurementRecord(-2), MeasurementRecord(-5)}, (2, 3, 0)),
            Detector({MeasurementRecord(-1), MeasurementRecord(-4)}, (2, 1, 0)),
            ShiftCoordinates((0, 0, 1)),
            GateLayer(
                {
                    RX(Qubit(Coord2D(0, 0))),
                    RZ(Qubit(Coord2D(0, 4))),
                    RX(Qubit(Coord2D(0, 6))),
                }
            ),
        ]
    ),
)

full_stage_many_rounds_not_using_ancilla = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=2,
        stabilisers=[
            [
                Stabiliser(stabiliser.paulis)
                for stabiliser in example_simultaneous_stabilisers[0]
            ]
        ],
        first_round_measurements=[
            MX(Qubit(Coord2D(0, 0))),
            MZ(Qubit(Coord2D(0, 4))),
            MX(Qubit(Coord2D(0, 6))),
        ],
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
        observable_definitions={0: [Qubit(Coord2D(0, 0))]},
        use_ancilla_qubits=False,
    ),
    expected_first_round=Circuit(
        [
            GateLayer(
                [
                    MX(Qubit(Coord2D(0, 0))),
                    MZ(Qubit(Coord2D(0, 4))),
                    MX(Qubit(Coord2D(0, 6))),
                ],
            ),
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 3))),
                    I(Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                MPP(
                    (
                        PauliX(Coord2D(3, 3)),
                        PauliX(Coord2D(3, 1)),
                        PauliX(Coord2D(1, 3)),
                        PauliX(Coord2D(1, 1)),
                    )
                )
            ),
            GateLayer(
                [
                    MPP((PauliZ(Coord2D(3, 3)), PauliZ(Coord2D(1, 3)))),
                    MPP((PauliZ(Coord2D(3, 1)), PauliZ(Coord2D(1, 1)))),
                ]
            ),
            Observable(0, [MeasurementRecord(-6)]),
        ]
    ),
    expected_remaining_rounds=Circuit(
        [
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 3))),
                    I(Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                MPP(
                    (
                        PauliX(Coord2D(3, 3)),
                        PauliX(Coord2D(3, 1)),
                        PauliX(Coord2D(1, 3)),
                        PauliX(Coord2D(1, 1)),
                    )
                )
            ),
            GateLayer(
                [
                    MPP((PauliZ(Coord2D(3, 3)), PauliZ(Coord2D(1, 3)))),
                    MPP((PauliZ(Coord2D(3, 1)), PauliZ(Coord2D(1, 1)))),
                ]
            ),
            Detector({MeasurementRecord(-3), MeasurementRecord(-6)}, (2, 2, 0)),
            Detector({MeasurementRecord(-2), MeasurementRecord(-5)}, (2, 3, 0)),
            Detector({MeasurementRecord(-1), MeasurementRecord(-4)}, (2, 1, 0)),
            ShiftCoordinates((0, 0, 1)),
            GateLayer(
                {
                    RX(Qubit(Coord2D(0, 0))),
                    RZ(Qubit(Coord2D(0, 4))),
                    RX(Qubit(Coord2D(0, 6))),
                }
            ),
        ]
    ),
)

full_stage_many_rounds_spaced_no_ancilla = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=2,
        stabilisers=[
            [Stabiliser(stabiliser.paulis) for stabiliser in stabiliser_set]
            for stabiliser_set in example_spaced_stabilisers
        ],
        first_round_measurements=[
            MX(Qubit(Coord2D(0, 0))),
            MZ(Qubit(Coord2D(0, 4))),
            MX(Qubit(Coord2D(0, 6))),
        ],
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
        observable_definitions={0: [Qubit(Coord2D(0, 0))]},
    ),
    expected_first_round=Circuit(
        [
            GateLayer(
                [
                    MX(Qubit(Coord2D(0, 0))),
                    MZ(Qubit(Coord2D(0, 4))),
                    MX(Qubit(Coord2D(0, 6))),
                ],
            ),
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 3))),
                    I(Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                MPP(
                    (
                        PauliX(Coord2D(3, 3)),
                        PauliX(Coord2D(3, 1)),
                        PauliX(Coord2D(1, 3)),
                        PauliX(Coord2D(1, 1)),
                    )
                )
            ),
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 3))),
                    I(Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                [
                    MPP((PauliZ(Coord2D(3, 3)), PauliZ(Coord2D(1, 3)))),
                    MPP((PauliZ(Coord2D(3, 1)), PauliZ(Coord2D(1, 1)))),
                ]
            ),
            Observable(0, [MeasurementRecord(-6)]),
        ]
    ),
    expected_remaining_rounds=Circuit(
        [
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 3))),
                    I(Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                MPP(
                    (
                        PauliX(Coord2D(3, 3)),
                        PauliX(Coord2D(3, 1)),
                        PauliX(Coord2D(1, 3)),
                        PauliX(Coord2D(1, 1)),
                    )
                )
            ),
            GateLayer(
                {
                    I(Qubit(Coord2D(3, 3))),
                    I(Qubit(Coord2D(3, 1))),
                    I(Qubit(Coord2D(1, 3))),
                    I(Qubit(Coord2D(1, 1))),
                }
            ),
            GateLayer(
                [
                    MPP((PauliZ(Coord2D(3, 3)), PauliZ(Coord2D(1, 3)))),
                    MPP((PauliZ(Coord2D(3, 1)), PauliZ(Coord2D(1, 1)))),
                ]
            ),
            Detector({MeasurementRecord(-3), MeasurementRecord(-6)}, (2, 2, 0)),
            Detector({MeasurementRecord(-2), MeasurementRecord(-5)}, (2, 3, 0)),
            Detector({MeasurementRecord(-1), MeasurementRecord(-4)}, (2, 1, 0)),
            ShiftCoordinates((0, 0, 1)),
            GateLayer(
                {
                    RX(Qubit(Coord2D(0, 0))),
                    RZ(Qubit(Coord2D(0, 4))),
                    RX(Qubit(Coord2D(0, 6))),
                }
            ),
        ]
    ),
)
