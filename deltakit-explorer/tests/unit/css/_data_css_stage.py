# (c) Copyright Riverlane 2020-2025.
from dataclasses import dataclass

from deltakit_circuit import (Circuit, Detector, GateLayer, MeasurementRecord,
                              Observable, PauliX, PauliZ, Qubit,
                              ShiftCoordinates)
from deltakit_circuit._basic_types import Coord2D, Coord2DDelta
from deltakit_circuit.gates import (CX, CZ, MPP, MX, MZ, RX, RZ, SWAP, H, I, X,
                                    Z)
from deltakit_explorer.codes._css._css_stage import CSSStage
from deltakit_explorer.codes._planar_code._rotated_planar_code import \
    RotatedPlanarCode
from deltakit_explorer.codes._stabiliser import Stabiliser

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


example_simultaneous_stabilisers = [
    [
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
            paulis=(None, None, PauliZ(Coord2D(3, 3)), PauliZ(Coord2D(1, 3))),
            ancilla_qubit=Qubit(Coord2D(2, 4)),
        ),
        Stabiliser(
            paulis=(PauliZ(Coord2D(3, 1)), PauliZ(Coord2D(1, 1)), None, None),
            ancilla_qubit=Qubit(Coord2D(2, 0)),
        ),
    ]
]
example_simultaneous_stabilisers_first_round_circuit = Circuit(
    [
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
    ]
)

example_spaced_stabilisers = [
    [
        Stabiliser(
            paulis=(
                PauliX(Coord2D(3, 3)),
                PauliX(Coord2D(3, 1)),
                PauliX(Coord2D(1, 3)),
                PauliX(Coord2D(1, 1)),
            ),
            ancilla_qubit=Qubit(Coord2D(2, 2)),
        ),
    ],
    [
        Stabiliser(
            paulis=(None, None, PauliZ(Coord2D(3, 3)), PauliZ(Coord2D(1, 3))),
            ancilla_qubit=Qubit(Coord2D(2, 4)),
        ),
        Stabiliser(
            paulis=(PauliZ(Coord2D(3, 1)), PauliZ(Coord2D(1, 1)), None, None),
            ancilla_qubit=Qubit(Coord2D(2, 0)),
        ),
    ],
]

measurements_and_observables_only_stage = CSSStageTestComponents(
    stage=CSSStage(
        first_round_measurements=[MZ(0)],
        observable_definitions={0: [Qubit(0)]},
    ),
)


resets_only_stage = CSSStageTestComponents(
    stage=CSSStage(
        final_round_resets=[RZ(1)],
    )
)


stabiliser_stage = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=1,
        stabilisers=example_simultaneous_stabilisers,
    ),
    expected_first_round=example_simultaneous_stabilisers_first_round_circuit,
    expected_remaining_rounds=Circuit(),
)

stabiliser_stage_spaced = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=1,
        stabilisers=example_spaced_stabilisers,
    ),
    expected_first_round=Circuit(
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
        ]
    ),
    expected_remaining_rounds=Circuit(),
)

stabiliser_meas_stage = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=1,
        stabilisers=example_simultaneous_stabilisers,
        first_round_measurements=[
            MX(Qubit(Coord2D(0, 0))),
            MZ(Qubit(Coord2D(0, 4))),
            MX(Qubit(Coord2D(0, 6))),
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
    expected_remaining_rounds=Circuit(),
    expected_measurements_as_stabilisers=(),
    expected_resets_as_stabilisers=(),
)

stabiliser_reset_stage = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=1,
        stabilisers=example_simultaneous_stabilisers,
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
    ),
    expected_first_round=Circuit(
        [
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
    expected_measurements_as_stabilisers=(),
)

stabiliser_reset_stage_many_rounds = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=4,
        stabilisers=example_simultaneous_stabilisers,
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
    ),
    expected_first_round=Circuit(
        [
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
        ]
    ),
    expected_remaining_rounds=Circuit(
        [
            Circuit(
                [
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
                            CZ(
                                Qubit(Coord2D(2, 0)),
                                Qubit(Coord2D(3, 1)),
                            ),
                        }
                    ),
                    GateLayer(
                        {
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(3, 1)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 0)),
                                Qubit(Coord2D(1, 1)),
                            ),
                        }
                    ),
                    GateLayer(
                        {
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(1, 3)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 4)),
                                Qubit(Coord2D(3, 3)),
                            ),
                        }
                    ),
                    GateLayer(
                        {
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(1, 1)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 4)),
                                Qubit(Coord2D(1, 3)),
                            ),
                        }
                    ),
                    GateLayer(
                        [
                            MX(Qubit(Coord2D(2, 2))),
                            MX(Qubit(Coord2D(2, 4))),
                            MX(Qubit(Coord2D(2, 0))),
                        ]
                    ),
                    Detector(
                        {MeasurementRecord(-3), MeasurementRecord(-6)},
                        (2, 2, 0),
                    ),
                    Detector(
                        {MeasurementRecord(-2), MeasurementRecord(-5)},
                        (2, 4, 0),
                    ),
                    Detector(
                        {MeasurementRecord(-1), MeasurementRecord(-4)},
                        (2, 0, 0),
                    ),
                    ShiftCoordinates((0, 0, 1)),
                ],
                iterations=3,
            ),
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

final_round_with_mpps_stage = CSSStageTestComponents(
    stage=CSSStage(
        stabilisers=example_simultaneous_stabilisers,
        num_rounds=1,
        first_round_measurements=[MPP([PauliZ(Coord2D(1, 1)), PauliZ(Coord2D(1, 3))])],
        observable_definitions={
            0: [MPP([PauliZ(Coord2D(1, 1)), PauliZ(Coord2D(1, 3))])]
        },
        use_ancilla_qubits=False,
    ),
    expected_first_round=Circuit(
        [
            GateLayer(MPP((PauliZ(Coord2D(1, 1)), PauliZ(Coord2D(1, 3))))),
            GateLayer(
                [
                    I(Coord2D(3, 3)),
                    I(Coord2D(3, 1)),
                    I(Coord2D(1, 3)),
                    I(Coord2D(1, 1)),
                ]
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
            Observable(0, [MeasurementRecord(-4)]),
        ]
    ),
    expected_remaining_rounds=Circuit(),
    expected_measurements_as_stabilisers=(
        Stabiliser([PauliZ(Coord2D(1, 1)), PauliZ(Coord2D(1, 3))]),
    ),
)

transv_h_stage = CSSStageTestComponents(
    stage=CSSStage(
        stabilisers=RotatedPlanarCode(2, 2).stabilisers,
        num_rounds=2,
        first_round_gates=[H(dq) for dq in RotatedPlanarCode(2, 2).data_qubits],
    )
)

transv_h_with_reset_stage = CSSStageTestComponents(
    stage=CSSStage(
        stabilisers=RotatedPlanarCode(2, 2).stabilisers,
        num_rounds=2,
        first_round_gates=[H(dq) for dq in RotatedPlanarCode(2, 2).data_qubits],
        final_round_resets=[RZ(Coord2D(-1, -1))],
    )
)

transv_swap_stage = CSSStageTestComponents(
    stage=CSSStage(
        stabilisers=RotatedPlanarCode(2, 2).stabilisers,
        num_rounds=2,
        first_round_gates=[
            SWAP(dq, dq.unique_identifier + Coord2DDelta(1, 1))
            for dq in RotatedPlanarCode(2, 2).data_qubits
        ],
    )
)

transv_swap_with_reset_stage = CSSStageTestComponents(
    stage=CSSStage(
        stabilisers=RotatedPlanarCode(2, 2).stabilisers,
        num_rounds=2,
        first_round_gates=[
            SWAP(dq, dq.unique_identifier + Coord2DDelta(1, 1))
            for dq in RotatedPlanarCode(2, 2).data_qubits
        ],
        final_round_resets=[RZ(Coord2D(-1, -1))],
    )
)

half_transv_h_stage = CSSStageTestComponents(
    stage=CSSStage(
        stabilisers=RotatedPlanarCode(2, 2).stabilisers,
        num_rounds=2,
        first_round_gates=[
            H(dq)
            for dq in RotatedPlanarCode(2, 2).data_qubits
            if dq.unique_identifier.y < 2
        ],
    )
)

data_x_stage = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=4,
        stabilisers=example_simultaneous_stabilisers,
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
        first_round_gates=[X(Coord2D(1, 1)), X(Coord2D(3, 3))],
    ),
    expected_first_round=Circuit(
        [
            GateLayer(
                {
                    X(Qubit(Coord2D(1, 1))),
                    X(Qubit(Coord2D(3, 3))),
                }
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
        ]
    ),
    expected_remaining_rounds=Circuit(
        [
            Circuit(
                [
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
                            CZ(
                                Qubit(Coord2D(2, 0)),
                                Qubit(Coord2D(3, 1)),
                            ),
                        }
                    ),
                    GateLayer(
                        {
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(3, 1)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 0)),
                                Qubit(Coord2D(1, 1)),
                            ),
                        }
                    ),
                    GateLayer(
                        {
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(1, 3)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 4)),
                                Qubit(Coord2D(3, 3)),
                            ),
                        }
                    ),
                    GateLayer(
                        {
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(1, 1)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 4)),
                                Qubit(Coord2D(1, 3)),
                            ),
                        }
                    ),
                    GateLayer(
                        [
                            MX(Qubit(Coord2D(2, 2))),
                            MX(Qubit(Coord2D(2, 4))),
                            MX(Qubit(Coord2D(2, 0))),
                        ]
                    ),
                    Detector(
                        {MeasurementRecord(-3), MeasurementRecord(-6)},
                        (2, 2, 0),
                    ),
                    Detector(
                        {MeasurementRecord(-2), MeasurementRecord(-5)},
                        (2, 4, 0),
                    ),
                    Detector(
                        {MeasurementRecord(-1), MeasurementRecord(-4)},
                        (2, 0, 0),
                    ),
                    ShiftCoordinates((0, 0, 1)),
                ],
                iterations=3,
            ),
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

data_z_stage = CSSStageTestComponents(
    stage=CSSStage(
        num_rounds=4,
        stabilisers=example_simultaneous_stabilisers,
        final_round_resets=[
            RX(Qubit(Coord2D(0, 0))),
            RZ(Qubit(Coord2D(0, 4))),
            RX(Qubit(Coord2D(0, 6))),
        ],
        first_round_gates=[Z(Coord2D(1, 1)), Z(Coord2D(3, 3))],
    ),
    expected_first_round=Circuit(
        [
            GateLayer(
                {
                    Z(Qubit(Coord2D(1, 1))),
                    Z(Qubit(Coord2D(3, 3))),
                }
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
        ]
    ),
    expected_remaining_rounds=Circuit(
        [
            Circuit(
                [
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
                            CZ(
                                Qubit(Coord2D(2, 0)),
                                Qubit(Coord2D(3, 1)),
                            ),
                        }
                    ),
                    GateLayer(
                        {
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(3, 1)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 0)),
                                Qubit(Coord2D(1, 1)),
                            ),
                        }
                    ),
                    GateLayer(
                        {
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(1, 3)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 4)),
                                Qubit(Coord2D(3, 3)),
                            ),
                        }
                    ),
                    GateLayer(
                        {
                            CX(
                                Qubit(Coord2D(2, 2)),
                                Qubit(Coord2D(1, 1)),
                            ),
                            CZ(
                                Qubit(Coord2D(2, 4)),
                                Qubit(Coord2D(1, 3)),
                            ),
                        }
                    ),
                    GateLayer(
                        [
                            MX(Qubit(Coord2D(2, 2))),
                            MX(Qubit(Coord2D(2, 4))),
                            MX(Qubit(Coord2D(2, 0))),
                        ]
                    ),
                    Detector(
                        {MeasurementRecord(-3), MeasurementRecord(-6)},
                        (2, 2, 0),
                    ),
                    Detector(
                        {MeasurementRecord(-2), MeasurementRecord(-5)},
                        (2, 4, 0),
                    ),
                    Detector(
                        {MeasurementRecord(-1), MeasurementRecord(-4)},
                        (2, 0, 0),
                    ),
                    ShiftCoordinates((0, 0, 1)),
                ],
                iterations=3,
            ),
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
