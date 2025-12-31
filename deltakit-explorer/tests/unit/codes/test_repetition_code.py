# (c) Copyright Riverlane 2020-2025.

import pytest
from deltakit_circuit import PauliX, PauliZ, Qubit
from deltakit_circuit._basic_types import Coord2D
from deltakit_circuit.gates import MX, MZ, RX, RZ, PauliBasis
from deltakit_explorer.codes._css._css_stage import CSSStage
from deltakit_explorer.codes._repetition_code import RepetitionCode
from deltakit_explorer.codes._stabiliser import Stabiliser


@pytest.mark.parametrize("stabiliser_type", [PauliBasis.X, PauliBasis.Z])
@pytest.mark.parametrize("distance", [1])
def test__check_distance_at_least_2_raises_error_with_distance_less_than_2(
    distance,
    stabiliser_type,
):
    with pytest.raises(
        ValueError,
        match="Code distance must be at least 2.",
    ):
        RepetitionCode(distance=distance, stabiliser_type=stabiliser_type)


@pytest.mark.parametrize("stabiliser_type", [PauliBasis.Y, "foo"])
def test__check_stabiliser_type_is_valid(
    stabiliser_type,
):
    with pytest.raises(
        ValueError,
        match=f"{stabiliser_type} is unsupported, only PauliBasis.X and "
        "PauliBasis.Z are allowed.",
    ):
        RepetitionCode(3, stabiliser_type)


@pytest.mark.parametrize("stabiliser_type", [PauliBasis.X, PauliBasis.Z])
@pytest.mark.parametrize("use_ancilla_qubits", [True, False])
@pytest.mark.parametrize("default_schedule", [True, False])
@pytest.mark.parametrize(
    "distance, odd_data_qubit_coords, expected_data_qubits",
    [
        (
            5,
            False,
            {
                Qubit(Coord2D(0, 0)),
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(4, 0)),
                Qubit(Coord2D(6, 0)),
                Qubit(Coord2D(8, 0)),
            },
        ),
        (
            3,
            False,
            {
                Qubit(Coord2D(0, 0)),
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(4, 0)),
            },
        ),
        (
            5,
            True,
            {
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(3, 0)),
                Qubit(Coord2D(5, 0)),
                Qubit(Coord2D(7, 0)),
                Qubit(Coord2D(9, 0)),
            },
        ),
        (
            3,
            True,
            {
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(3, 0)),
                Qubit(Coord2D(5, 0)),
            },
        ),
    ],
)
def test__data_qubits_gives_correct_set_of_qubits(
    stabiliser_type,
    use_ancilla_qubits,
    default_schedule,
    distance,
    odd_data_qubit_coords,
    expected_data_qubits,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits,
        default_schedule=default_schedule,
        odd_data_qubit_coords=odd_data_qubit_coords,
    )
    assert code._data_qubits == expected_data_qubits


@pytest.mark.parametrize("stabiliser_type", [PauliBasis.X, PauliBasis.Z])
@pytest.mark.parametrize("use_ancilla_qubits", [True, False])
@pytest.mark.parametrize("use_looping_stabiliser", [False, True])
@pytest.mark.parametrize("default_schedule", [True, False])
@pytest.mark.parametrize(
    "distance, odd_data_qubit_coords, expected_ancilla_qubits",
    [
        (
            5,
            False,
            {
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(3, 0)),
                Qubit(Coord2D(5, 0)),
                Qubit(Coord2D(7, 0)),
            },
        ),
        (
            3,
            False,
            {
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(3, 0)),
            },
        ),
        (
            5,
            True,
            {
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(4, 0)),
                Qubit(Coord2D(6, 0)),
                Qubit(Coord2D(8, 0)),
            },
        ),
        (
            3,
            True,
            {
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(4, 0)),
            },
        ),
    ],
)
def test__stabiliser_ancilla_qubits_gives_correct_set_of_qubits(
    stabiliser_type,
    use_ancilla_qubits,
    use_looping_stabiliser,
    default_schedule,
    distance,
    odd_data_qubit_coords,
    expected_ancilla_qubits,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits,
        use_looping_stabiliser=use_looping_stabiliser,
        default_schedule=default_schedule,
        odd_data_qubit_coords=odd_data_qubit_coords,
    )
    if use_looping_stabiliser:
        if odd_data_qubit_coords:
            assert code._stabiliser_ancilla_qubits == expected_ancilla_qubits | {
                Qubit(Coord2D(0, 0))
            }
        else:
            assert code._stabiliser_ancilla_qubits == expected_ancilla_qubits | {
                Qubit(Coord2D(2 * distance - 1, 0))
            }
    else:
        assert code._stabiliser_ancilla_qubits == expected_ancilla_qubits


@pytest.mark.parametrize("stabiliser_type", [PauliBasis.X, PauliBasis.Z])
@pytest.mark.parametrize("use_looping_stabiliser", [False, True])
@pytest.mark.parametrize("default_schedule", [True, False])
@pytest.mark.parametrize(
    "distance, use_ancilla_qubits, odd_data_qubit_coords, expected_ancillas",
    [
        (
            3,
            True,
            False,
            {
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(3, 0)),
            },
        ),
        (
            5,
            True,
            False,
            {
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(3, 0)),
                Qubit(Coord2D(5, 0)),
                Qubit(Coord2D(7, 0)),
            },
        ),
        (
            3,
            False,
            False,
            set(),
        ),
        (
            5,
            False,
            False,
            set(),
        ),
        (
            3,
            True,
            True,
            {
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(4, 0)),
            },
        ),
        (
            5,
            True,
            True,
            {
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(4, 0)),
                Qubit(Coord2D(6, 0)),
                Qubit(Coord2D(8, 0)),
            },
        ),
        (
            3,
            False,
            True,
            set(),
        ),
        (
            5,
            False,
            True,
            set(),
        ),
    ],
)
def test__ancilla_qubits_gives_correct_qubits(
    stabiliser_type,
    use_looping_stabiliser,
    default_schedule,
    distance,
    use_ancilla_qubits,
    odd_data_qubit_coords,
    expected_ancillas,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits,
        use_looping_stabiliser=use_looping_stabiliser,
        default_schedule=default_schedule,
        odd_data_qubit_coords=odd_data_qubit_coords,
    )
    if use_looping_stabiliser and use_ancilla_qubits:
        if odd_data_qubit_coords:
            assert code._ancilla_qubits == expected_ancillas | {Qubit(Coord2D(0, 0))}
        else:
            assert code._ancilla_qubits == expected_ancillas | {
                Qubit(Coord2D(2 * distance - 1, 0))
            }
    else:
        assert code._ancilla_qubits == expected_ancillas


@pytest.mark.parametrize("stabiliser_type", [PauliBasis.X, PauliBasis.Z])
@pytest.mark.parametrize("use_looping_stabiliser", [False, True])
@pytest.mark.parametrize("default_schedule", [True, False])
@pytest.mark.parametrize(
    "distance, use_ancilla_qubits, odd_data_qubit_coords, expected_qubits",
    [
        (
            3,
            True,
            False,
            {
                Qubit(Coord2D(0, 0)),
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(3, 0)),
                Qubit(Coord2D(4, 0)),
            },
        ),
        (
            3,
            False,
            False,
            {
                Qubit(Coord2D(0, 0)),
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(4, 0)),
            },
        ),
        (
            5,
            True,
            False,
            {
                Qubit(Coord2D(0, 0)),
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(3, 0)),
                Qubit(Coord2D(4, 0)),
                Qubit(Coord2D(5, 0)),
                Qubit(Coord2D(6, 0)),
                Qubit(Coord2D(7, 0)),
                Qubit(Coord2D(8, 0)),
            },
        ),
        (
            5,
            False,
            False,
            {
                Qubit(Coord2D(0, 0)),
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(4, 0)),
                Qubit(Coord2D(6, 0)),
                Qubit(Coord2D(8, 0)),
            },
        ),
        (
            3,
            True,
            True,
            {
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(3, 0)),
                Qubit(Coord2D(4, 0)),
                Qubit(Coord2D(5, 0)),
            },
        ),
        (
            3,
            False,
            True,
            {
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(3, 0)),
                Qubit(Coord2D(5, 0)),
            },
        ),
        (
            5,
            True,
            True,
            {
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(2, 0)),
                Qubit(Coord2D(3, 0)),
                Qubit(Coord2D(4, 0)),
                Qubit(Coord2D(5, 0)),
                Qubit(Coord2D(6, 0)),
                Qubit(Coord2D(7, 0)),
                Qubit(Coord2D(8, 0)),
                Qubit(Coord2D(9, 0)),
            },
        ),
        (
            5,
            False,
            True,
            {
                Qubit(Coord2D(1, 0)),
                Qubit(Coord2D(3, 0)),
                Qubit(Coord2D(5, 0)),
                Qubit(Coord2D(7, 0)),
                Qubit(Coord2D(9, 0)),
            },
        ),
    ],
)
def test_property_qubits_is_as_expected(
    stabiliser_type,
    use_looping_stabiliser,
    default_schedule,
    distance,
    use_ancilla_qubits,
    odd_data_qubit_coords,
    expected_qubits,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits,
        use_looping_stabiliser,
        odd_data_qubit_coords=odd_data_qubit_coords,
        default_schedule=default_schedule,
    )
    if use_looping_stabiliser and use_ancilla_qubits:
        if odd_data_qubit_coords:
            assert code.qubits == expected_qubits | {Qubit(Coord2D(0, 0))}
        else:
            assert code.qubits == expected_qubits | {
                Qubit(Coord2D(2 * distance - 1, 0))
            }
    else:
        assert code.qubits == expected_qubits


@pytest.mark.parametrize("use_ancilla_qubits", [True, False])
@pytest.mark.parametrize("default_schedule", [True, False])
@pytest.mark.parametrize(
    "distance, stabiliser_type, odd_data_qubit_coords, x_logicals_expected,"
    "z_logicals_expected",
    [
        (
            5,
            PauliBasis.Z,
            False,
            (
                {
                    PauliX(Coord2D(0, 0)),
                    PauliX(Coord2D(2, 0)),
                    PauliX(Coord2D(4, 0)),
                    PauliX(Coord2D(6, 0)),
                    PauliX(Coord2D(8, 0)),
                },
            ),
            (
                {
                    PauliZ(Coord2D(0, 0)),
                },
            ),
        ),
        (
            3,
            PauliBasis.X,
            False,
            (
                {
                    PauliX(Coord2D(0, 0)),
                },
            ),
            (
                {
                    PauliZ(Coord2D(0, 0)),
                    PauliZ(Coord2D(2, 0)),
                    PauliZ(Coord2D(4, 0)),
                },
            ),
        ),
        (
            5,
            PauliBasis.Z,
            True,
            (
                {
                    PauliX(Coord2D(1, 0)),
                    PauliX(Coord2D(3, 0)),
                    PauliX(Coord2D(5, 0)),
                    PauliX(Coord2D(7, 0)),
                    PauliX(Coord2D(9, 0)),
                },
            ),
            (
                {
                    PauliZ(Coord2D(1, 0)),
                },
            ),
        ),
        (
            3,
            PauliBasis.X,
            True,
            (
                {
                    PauliX(Coord2D(1, 0)),
                },
            ),
            (
                {
                    PauliZ(Coord2D(1, 0)),
                    PauliZ(Coord2D(3, 0)),
                    PauliZ(Coord2D(5, 0)),
                },
            ),
        ),
    ],
)
def test_logical_operators(
    use_ancilla_qubits,
    default_schedule,
    distance,
    stabiliser_type,
    odd_data_qubit_coords,
    x_logicals_expected,
    z_logicals_expected,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits,
        default_schedule=default_schedule,
        odd_data_qubit_coords=odd_data_qubit_coords,
    )
    assert code._x_logical_operators == x_logicals_expected
    assert code._z_logical_operators == z_logicals_expected


d_5_z_stabilisers = (
    (
        Stabiliser([PauliZ(Coord2D(0, 0)), PauliZ(Coord2D(2, 0))], Coord2D(1, 0)),
        Stabiliser([PauliZ(Coord2D(2, 0)), PauliZ(Coord2D(4, 0))], Coord2D(3, 0)),
        Stabiliser([PauliZ(Coord2D(4, 0)), PauliZ(Coord2D(6, 0))], Coord2D(5, 0)),
        Stabiliser([PauliZ(Coord2D(6, 0)), PauliZ(Coord2D(8, 0))], Coord2D(7, 0)),
    ),
)
d_5_z_stabilisers_odd_data = (
    (
        Stabiliser([PauliZ(Coord2D(1, 0)), PauliZ(Coord2D(3, 0))], Coord2D(2, 0)),
        Stabiliser([PauliZ(Coord2D(3, 0)), PauliZ(Coord2D(5, 0))], Coord2D(4, 0)),
        Stabiliser([PauliZ(Coord2D(5, 0)), PauliZ(Coord2D(7, 0))], Coord2D(6, 0)),
        Stabiliser([PauliZ(Coord2D(7, 0)), PauliZ(Coord2D(9, 0))], Coord2D(8, 0)),
    ),
)
d_5_z_stabilisers_reversed = (
    (
        Stabiliser([PauliZ(Coord2D(2, 0)), PauliZ(Coord2D(0, 0))], Coord2D(1, 0)),
        Stabiliser([PauliZ(Coord2D(4, 0)), PauliZ(Coord2D(2, 0))], Coord2D(3, 0)),
        Stabiliser([PauliZ(Coord2D(6, 0)), PauliZ(Coord2D(4, 0))], Coord2D(5, 0)),
        Stabiliser([PauliZ(Coord2D(8, 0)), PauliZ(Coord2D(6, 0))], Coord2D(7, 0)),
    ),
)
d_5_z_stabilisers_odd_data_reversed = (
    (
        Stabiliser([PauliZ(Coord2D(3, 0)), PauliZ(Coord2D(1, 0))], Coord2D(2, 0)),
        Stabiliser([PauliZ(Coord2D(5, 0)), PauliZ(Coord2D(3, 0))], Coord2D(4, 0)),
        Stabiliser([PauliZ(Coord2D(7, 0)), PauliZ(Coord2D(5, 0))], Coord2D(6, 0)),
        Stabiliser([PauliZ(Coord2D(9, 0)), PauliZ(Coord2D(7, 0))], Coord2D(8, 0)),
    ),
)
d_5_z_stabilisers_no_ancilla = (
    (
        Stabiliser([PauliZ(Coord2D(0, 0)), PauliZ(Coord2D(2, 0))], Coord2D(1, 0)),
        Stabiliser([PauliZ(Coord2D(4, 0)), PauliZ(Coord2D(6, 0))], Coord2D(5, 0)),
        Stabiliser([PauliZ(Coord2D(2, 0)), PauliZ(Coord2D(4, 0))], Coord2D(3, 0)),
        Stabiliser([PauliZ(Coord2D(6, 0)), PauliZ(Coord2D(8, 0))], Coord2D(7, 0)),
    ),
)
d_5_z_stabilisers_odd_data_no_ancilla = (
    (
        Stabiliser([PauliZ(Coord2D(1, 0)), PauliZ(Coord2D(3, 0))], Coord2D(2, 0)),
        Stabiliser([PauliZ(Coord2D(5, 0)), PauliZ(Coord2D(7, 0))], Coord2D(6, 0)),
        Stabiliser([PauliZ(Coord2D(3, 0)), PauliZ(Coord2D(5, 0))], Coord2D(4, 0)),
        Stabiliser([PauliZ(Coord2D(7, 0)), PauliZ(Coord2D(9, 0))], Coord2D(8, 0)),
    ),
)
d_5_z_stabilisers_no_ancilla_reversed = (
    (
        Stabiliser([PauliZ(Coord2D(2, 0)), PauliZ(Coord2D(0, 0))], Coord2D(1, 0)),
        Stabiliser([PauliZ(Coord2D(6, 0)), PauliZ(Coord2D(4, 0))], Coord2D(5, 0)),
        Stabiliser([PauliZ(Coord2D(4, 0)), PauliZ(Coord2D(2, 0))], Coord2D(3, 0)),
        Stabiliser([PauliZ(Coord2D(8, 0)), PauliZ(Coord2D(6, 0))], Coord2D(7, 0)),
    ),
)
d_5_z_stabilisers_odd_data_no_ancilla_reversed = (
    (
        Stabiliser([PauliZ(Coord2D(3, 0)), PauliZ(Coord2D(1, 0))], Coord2D(2, 0)),
        Stabiliser([PauliZ(Coord2D(7, 0)), PauliZ(Coord2D(5, 0))], Coord2D(6, 0)),
        Stabiliser([PauliZ(Coord2D(5, 0)), PauliZ(Coord2D(3, 0))], Coord2D(4, 0)),
        Stabiliser([PauliZ(Coord2D(9, 0)), PauliZ(Coord2D(7, 0))], Coord2D(8, 0)),
    ),
)
d_3_x_stabilisers = (
    (
        Stabiliser([PauliX(Coord2D(0, 0)), PauliX(Coord2D(2, 0))], Coord2D(1, 0)),
        Stabiliser([PauliX(Coord2D(2, 0)), PauliX(Coord2D(4, 0))], Coord2D(3, 0)),
    ),
)
d_3_x_stabilisers_odd_data = (
    (
        Stabiliser([PauliX(Coord2D(1, 0)), PauliX(Coord2D(3, 0))], Coord2D(2, 0)),
        Stabiliser([PauliX(Coord2D(3, 0)), PauliX(Coord2D(5, 0))], Coord2D(4, 0)),
    ),
)
d_3_x_stabilisers_reversed = (
    (
        Stabiliser([PauliX(Coord2D(2, 0)), PauliX(Coord2D(0, 0))], Coord2D(1, 0)),
        Stabiliser([PauliX(Coord2D(4, 0)), PauliX(Coord2D(2, 0))], Coord2D(3, 0)),
    ),
)
d_3_x_stabilisers_odd_data_reversed = (
    (
        Stabiliser([PauliX(Coord2D(3, 0)), PauliX(Coord2D(1, 0))], Coord2D(2, 0)),
        Stabiliser([PauliX(Coord2D(5, 0)), PauliX(Coord2D(3, 0))], Coord2D(4, 0)),
    ),
)
d_3_x_stabilisers_looped = (
    (
        Stabiliser([PauliX(Coord2D(4, 0)), PauliX(Coord2D(0, 0))], Coord2D(5, 0)),
        Stabiliser([PauliX(Coord2D(0, 0)), PauliX(Coord2D(2, 0))], Coord2D(1, 0)),
        Stabiliser([PauliX(Coord2D(2, 0)), PauliX(Coord2D(4, 0))], Coord2D(3, 0)),
    ),
)
d_3_x_stabilisers_odd_data_looped = (
    (
        Stabiliser([PauliX(Coord2D(5, 0)), PauliX(Coord2D(1, 0))], Coord2D(0, 0)),
        Stabiliser([PauliX(Coord2D(1, 0)), PauliX(Coord2D(3, 0))], Coord2D(2, 0)),
        Stabiliser([PauliX(Coord2D(3, 0)), PauliX(Coord2D(5, 0))], Coord2D(4, 0)),
    ),
)
d_3_x_stabilisers_looped_reversed = (
    (
        Stabiliser([PauliX(Coord2D(0, 0)), PauliX(Coord2D(4, 0))], Coord2D(5, 0)),
        Stabiliser([PauliX(Coord2D(2, 0)), PauliX(Coord2D(0, 0))], Coord2D(1, 0)),
        Stabiliser([PauliX(Coord2D(4, 0)), PauliX(Coord2D(2, 0))], Coord2D(3, 0)),
    ),
)
d_3_x_stabilisers_odd_data_looped_reversed = (
    (
        Stabiliser([PauliX(Coord2D(1, 0)), PauliX(Coord2D(5, 0))], Coord2D(0, 0)),
        Stabiliser([PauliX(Coord2D(3, 0)), PauliX(Coord2D(1, 0))], Coord2D(2, 0)),
        Stabiliser([PauliX(Coord2D(5, 0)), PauliX(Coord2D(3, 0))], Coord2D(4, 0)),
    ),
)


@pytest.mark.parametrize(
    "distance, stabiliser_type, use_ancilla_qubits, use_looping_stabiliser, "
    "odd_data_qubit_coords, default_schedule, expected_stabilisers",
    [
        (
            5,
            PauliBasis.Z,
            True,
            False,
            False,
            True,
            d_5_z_stabilisers,
        ),
        (
            5,
            PauliBasis.Z,
            True,
            False,
            True,
            True,
            d_5_z_stabilisers_odd_data,
        ),
        (
            5,
            PauliBasis.Z,
            True,
            False,
            False,
            False,
            d_5_z_stabilisers_reversed,
        ),
        (
            5,
            PauliBasis.Z,
            True,
            False,
            True,
            False,
            d_5_z_stabilisers_odd_data_reversed,
        ),
        (
            5,
            PauliBasis.Z,
            False,
            False,
            False,
            True,
            d_5_z_stabilisers_no_ancilla,
        ),
        (
            5,
            PauliBasis.Z,
            False,
            False,
            True,
            True,
            d_5_z_stabilisers_odd_data_no_ancilla,
        ),
        (
            5,
            PauliBasis.Z,
            False,
            False,
            False,
            False,
            d_5_z_stabilisers_no_ancilla_reversed,
        ),
        (
            5,
            PauliBasis.Z,
            False,
            False,
            True,
            False,
            d_5_z_stabilisers_odd_data_no_ancilla_reversed,
        ),
        (
            3,
            PauliBasis.X,
            True,
            False,
            False,
            True,
            d_3_x_stabilisers,
        ),
        (
            3,
            PauliBasis.X,
            True,
            False,
            True,
            True,
            d_3_x_stabilisers_odd_data,
        ),
        (
            3,
            PauliBasis.X,
            True,
            False,
            False,
            False,
            d_3_x_stabilisers_reversed,
        ),
        (
            3,
            PauliBasis.X,
            True,
            False,
            True,
            False,
            d_3_x_stabilisers_odd_data_reversed,
        ),
        (
            3,
            PauliBasis.X,
            True,
            True,
            False,
            True,
            d_3_x_stabilisers_looped,
        ),
        (
            3,
            PauliBasis.X,
            True,
            True,
            True,
            True,
            d_3_x_stabilisers_odd_data_looped,
        ),
        (
            3,
            PauliBasis.X,
            True,
            True,
            False,
            False,
            d_3_x_stabilisers_looped_reversed,
        ),
        (
            3,
            PauliBasis.X,
            True,
            True,
            True,
            False,
            d_3_x_stabilisers_odd_data_looped_reversed,
        ),
    ],
)
def test_stabilisers(
    distance,
    stabiliser_type,
    use_ancilla_qubits,
    use_looping_stabiliser,
    odd_data_qubit_coords,
    default_schedule,
    expected_stabilisers,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits=use_ancilla_qubits,
        use_looping_stabiliser=use_looping_stabiliser,
        default_schedule=default_schedule,
        odd_data_qubit_coords=odd_data_qubit_coords,
    )
    assert code.stabilisers == expected_stabilisers


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("stabiliser_type", [PauliBasis.X, PauliBasis.Z])
@pytest.mark.parametrize("use_ancilla_qubits", [True, False])
@pytest.mark.parametrize("use_looping_stabiliser", [False, True])
@pytest.mark.parametrize("odd_data_qubit_coords", [False, True])
@pytest.mark.parametrize("default_schedule", [True, False])
def test_encode_logical_zeroes_returns_expected_stage(
    distance,
    stabiliser_type,
    use_ancilla_qubits,
    use_looping_stabiliser,
    odd_data_qubit_coords,
    default_schedule,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits,
        use_looping_stabiliser=use_looping_stabiliser,
        default_schedule=default_schedule,
        odd_data_qubit_coords=odd_data_qubit_coords,
    )
    assert code.encode_logical_zeroes() == CSSStage(
        final_round_resets=[RZ(qubit) for qubit in code._data_qubits]
    )


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("stabiliser_type", [PauliBasis.X, PauliBasis.Z])
@pytest.mark.parametrize("use_ancilla_qubits", [True, False])
@pytest.mark.parametrize("use_looping_stabiliser", [False, True])
@pytest.mark.parametrize("odd_data_qubit_coords", [False, True])
@pytest.mark.parametrize("default_schedule", [True, False])
def test_encode_logical_pluses_returns_expected_stage(
    distance,
    stabiliser_type,
    use_ancilla_qubits,
    use_looping_stabiliser,
    odd_data_qubit_coords,
    default_schedule,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits,
        use_looping_stabiliser,
        odd_data_qubit_coords=odd_data_qubit_coords,
        default_schedule=default_schedule,
    )
    assert code.encode_logical_pluses() == CSSStage(
        final_round_resets=[RX(qubit) for qubit in code._data_qubits]
    )


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("stabiliser_type", [PauliBasis.X, PauliBasis.Z])
@pytest.mark.parametrize("use_ancilla_qubits", [True, False])
@pytest.mark.parametrize("use_looping_stabiliser", [True, False])
@pytest.mark.parametrize("odd_data_qubit_coords", [False, True])
@pytest.mark.parametrize("default_schedule", [True, False])
def test_measure_stabilisers_returns_expected_stage(
    distance,
    stabiliser_type,
    use_ancilla_qubits,
    use_looping_stabiliser,
    odd_data_qubit_coords,
    default_schedule,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits,
        use_looping_stabiliser,
        odd_data_qubit_coords=odd_data_qubit_coords,
        default_schedule=default_schedule,
    )
    num_rounds = 5
    expected_stage = CSSStage(
        num_rounds=num_rounds,
        stabilisers=code.stabilisers,
        use_ancilla_qubits=use_ancilla_qubits,
    )
    assert code.measure_stabilisers(num_rounds) == expected_stage


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("stabiliser_type", [PauliBasis.X, PauliBasis.Z])
@pytest.mark.parametrize("use_ancilla_qubits", [True, False])
@pytest.mark.parametrize("use_looping_stabiliser", [True, False])
@pytest.mark.parametrize("odd_data_qubit_coords", [False, True])
@pytest.mark.parametrize("default_schedule", [True, False])
def test_measure_z_logicals_returns_expected_stage(
    distance,
    stabiliser_type,
    use_ancilla_qubits,
    use_looping_stabiliser,
    odd_data_qubit_coords,
    default_schedule,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits,
        use_looping_stabiliser=use_looping_stabiliser,
        odd_data_qubit_coords=odd_data_qubit_coords,
        default_schedule=default_schedule,
    )
    expected_stage = CSSStage(
        first_round_measurements=[MZ(qubit) for qubit in code._data_qubits],
        observable_definitions={
            0: [pauli.qubit for pauli in code._z_logical_operators[0]]
        },
    )
    assert code.measure_z_logicals() == expected_stage


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("stabiliser_type", [PauliBasis.X, PauliBasis.Z])
@pytest.mark.parametrize("use_ancilla_qubits", [True, False])
@pytest.mark.parametrize("use_looping_stabiliser", [True, False])
@pytest.mark.parametrize("odd_data_qubit_coords", [False, True])
@pytest.mark.parametrize("default_schedule", [True, False])
def test_measure_x_logicals_returns_expected_stage(
    distance,
    stabiliser_type,
    use_ancilla_qubits,
    use_looping_stabiliser,
    odd_data_qubit_coords,
    default_schedule,
):
    code = RepetitionCode(
        distance,
        stabiliser_type,
        use_ancilla_qubits,
        use_looping_stabiliser=use_looping_stabiliser,
        odd_data_qubit_coords=odd_data_qubit_coords,
        default_schedule=default_schedule,
    )
    expected_stage = CSSStage(
        first_round_measurements=[MX(qubit) for qubit in code._data_qubits],
        observable_definitions={
            0: [pauli.qubit for pauli in code._x_logical_operators[0]]
        },
    )
    assert code.measure_x_logicals() == expected_stage
