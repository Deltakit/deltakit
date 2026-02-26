# (c) Copyright Riverlane 2020-2025.
import re
from copy import copy, deepcopy
from importlib.metadata import version

import pytest
import stim
from packaging.version import Version

from deltakit_circuit import (
    Circuit,
    Detector,
    GateLayer,
    InvertiblePauliX,
    InvertiblePauliY,
    InvertiblePauliZ,
    MeasurementPauliProduct,
    MeasurementRecord,
    NoiseLayer,
    Observable,
    PauliX,
    PauliY,
    PauliZ,
    Qubit,
    ShiftCoordinates,
    gates,
)
from deltakit_circuit._gate_layer import DuplicateQubitError
from deltakit_circuit.noise_channels import PauliXError

CURRENT_STIM_VERSION = Version(version("stim"))
STIM_VERSION_V1_13_0 = Version("1.13.0")


@pytest.fixture
def empty_layer() -> GateLayer:
    return GateLayer()


@pytest.fixture
def gate_layer() -> GateLayer:
    layer = GateLayer()
    layer.add_gates(gates.X(Qubit(i)) for i in range(3))
    return layer


@pytest.mark.parametrize(
    "gates",
    [[gates.X(Qubit(i)) for i in range(3)], list(gates.CX.from_consecutive(range(4)))],
)
def test_gate_layer_repr_matches_expected_representation(empty_layer: GateLayer, gates):
    empty_layer.add_gates(gates)
    representation = repr(empty_layer)
    assert representation.startswith("GateLayer([\n")
    assert all(str(gate) in representation for gate in empty_layer.gates)
    assert representation.endswith("])")


def test_gate_layer_can_be_initialised_with_a_single_gate():
    gate = gates.X(Qubit(0))
    layer = GateLayer(gate)
    assert gate in layer.gates


@pytest.mark.parametrize(
    "gates",
    [[gates.X(Qubit(4)), gates.Y(Qubit(2))], (gates.H(Qubit(i)) for i in range(2))],
)
def test_gate_layer_can_be_initialised_with_a_list_of_gates(gates):
    layer = GateLayer(gates)
    assert all(gate in layer.gates for gate in gates)


@pytest.mark.parametrize("gate_class", gates.ONE_QUBIT_GATES)
def test_adding_a_single_qubit_gate_to_a_layer_puts_it_in_the_layer(
    gate_class: type[gates._OneQubitCliffordGate], empty_layer
):
    gate = gate_class(Qubit(0))
    empty_layer.add_gates(gate)
    assert gate in empty_layer.gates


@pytest.mark.parametrize("gate_class", gates.TWO_QUBIT_GATES)
def test_adding_two_qubit_gate_to_a_layer_puts_it_in_the_layer(
    gate_class: type[gates._TwoQubitGate], empty_layer
):
    gate = gate_class(Qubit(0), Qubit(1))
    empty_layer.add_gates(gate)
    assert gate in empty_layer.gates


@pytest.mark.parametrize("gate_class", gates.RESET_GATES)
def test_adding_reset_gate_to_a_layer_puts_it_in_the_layer(
    gate_class: type[gates._ResetGate], empty_layer
):
    gate = gate_class(Qubit(0))
    empty_layer.add_gates(gate)
    assert gate in empty_layer.gates


@pytest.mark.parametrize("gate_class", gates.MEASUREMENT_GATES - {gates.MPP})
@pytest.mark.parametrize("invert", [True, False])
def test_adding_measurement_gate_to_a_layer_puts_it_in_the_layer(
    gate_class: type[gates._MeasurementGate], invert, empty_layer
):
    gate = gate_class(Qubit(0), invert=invert)
    empty_layer.add_gates(gate)
    assert gate in empty_layer.gates


@pytest.mark.parametrize(
    "gate_generator",
    [
        (gates.X(Qubit(index)) for index in range(4)),
        (gates.RX(Qubit(index)) for index in range(2)),
        (gates.CX(Qubit(i), Qubit(i + 1)) for i in range(0, 4, 2)),
    ],
)
def test_adding_gates_from_a_generator_puts_each_in_the_layer(
    empty_layer, gate_generator
):
    empty_layer.add_gates(gate_generator)
    assert all(gate in empty_layer.gates for gate in gate_generator)


def test_adding_multiple_measurements_to_layer_keeps_the_correct_order(
    empty_layer: GateLayer,
):
    measurement_gates = (gates.MX(Qubit(0)), gates.MY(Qubit(1)), gates.MRZ(Qubit(2)))
    empty_layer.add_gates(measurement_gates)
    assert empty_layer.measurement_gates == measurement_gates


@pytest.mark.parametrize(
    "qubit_identifier",
    [
        InvertiblePauliX(Qubit(0)),
        InvertiblePauliY(Qubit(1)),
        InvertiblePauliZ(Qubit(2)),
        ~InvertiblePauliX(Qubit(0)),
        ~InvertiblePauliY(Qubit(1)),
        ~InvertiblePauliZ(Qubit(2)),
        MeasurementPauliProduct((PauliX(Qubit(1)), PauliY(Qubit(2)))),
        MeasurementPauliProduct((~InvertiblePauliX(Qubit(0)), PauliZ(Qubit(1)))),
    ],
)
def test_adding_mpp_gate_to_a_layer_adds_it_to_the_layer(qubit_identifier, empty_layer):
    gate = gates.MPP(qubit_identifier)
    empty_layer.add_gates(gate)
    assert gate in empty_layer.gates


def test_adding_a_gate_increases_the_number_of_gates_in_the_layer(empty_layer):
    empty_layer.add_gates(gates.H(Qubit(0)))
    assert len(empty_layer.gates) == 1


@pytest.mark.parametrize("single_qubit_gate_class", gates.ONE_QUBIT_GATES)
def test_error_is_raised_when_adding_two_single_qubit_gates_on_the_same_qubit(
    single_qubit_gate_class: type[gates._OneQubitCliffordGate], empty_layer
):
    qubit = Qubit(0)
    empty_layer.add_gates(gates.H(qubit))
    new_gate = single_qubit_gate_class(qubit)
    with pytest.raises(
        DuplicateQubitError,
        match=re.escape(
            f"For gate {new_gate}: qubits {{Qubit(0)}} "
            "were identified as being duplicates in the "
            "layer."
        ),
    ):
        empty_layer.add_gates(new_gate)


@pytest.mark.parametrize("two_qubit_gate_class", gates.TWO_QUBIT_GATES)
def test_error_is_raised_when_adding_two_qubit_gate_with_one_of_qubits_is_in_layer(
    two_qubit_gate_class: type[gates._TwoQubitGate], empty_layer
):
    qubit = Qubit(4)
    empty_layer.add_gates(gates.H(qubit))
    new_gate = two_qubit_gate_class(Qubit(0), qubit)
    with pytest.raises(
        DuplicateQubitError,
        match=re.escape(
            f"For gate {new_gate}: qubits {{Qubit(4)}} "
            "were identified as being duplicates in the "
            "layer."
        ),
    ):
        empty_layer.add_gates(new_gate)


@pytest.mark.parametrize("reset_gate_class", gates.RESET_GATES)
def test_error_is_raised_when_adding_reset_gate_to_a_layer_which_already_uses_that_qubit(
    reset_gate_class: type[gates._ResetGate], empty_layer
):
    qubit = Qubit(2)
    empty_layer.add_gates(gates.H(qubit))
    new_gate = reset_gate_class(qubit)
    with pytest.raises(
        DuplicateQubitError,
        match=re.escape(
            f"For gate {new_gate}: qubits {{Qubit(2)}} "
            "were identified as being duplicates in the "
            "layer."
        ),
    ):
        empty_layer.add_gates(new_gate)


@pytest.mark.parametrize(
    "measurement_gate_class", gates.MEASUREMENT_GATES - {gates.MPP}
)
@pytest.mark.parametrize("invert", [True, False])
def test_error_is_raised_when_adding_a_measurement_gate_to_a_layer_which_already_uses_that_qubit(
    measurement_gate_class: type[gates._MeasurementGate], invert, empty_layer
):
    qubit = Qubit(0)
    empty_layer.add_gates(gates.H(qubit))
    new_gate = measurement_gate_class(qubit, invert=invert)
    with pytest.raises(
        DuplicateQubitError,
        match=re.escape(
            f"For gate {new_gate}: qubits {{Qubit(0)}} "
            "were identified as being duplicates in the "
            "layer."
        ),
    ):
        empty_layer.add_gates(new_gate)


@pytest.mark.parametrize(
    "mpp_qubit_id",
    [
        ~InvertiblePauliZ(Qubit(0)),
        MeasurementPauliProduct(
            [InvertiblePauliX(Qubit(0)), InvertiblePauliY(Qubit(2))]
        ),
        MeasurementPauliProduct(
            [~InvertiblePauliY(Qubit(1)), InvertiblePauliZ(Qubit(0))]
        ),
    ],
)
def test_error_is_raised_when_adding_an_mpp_gate_on_a_qubit_that_is_already_in_use(
    empty_layer, mpp_qubit_id
):
    empty_layer.add_gates(gates.H(Qubit(0)))
    new_gate = gates.MPP(mpp_qubit_id)
    with pytest.raises(
        DuplicateQubitError,
        match=re.escape(
            f"For gate {new_gate}: qubits {{Qubit(0)}} "
            "were identified as being duplicates in the "
            "layer."
        ),
    ):
        empty_layer.add_gates(new_gate)


def test_error_is_raised_with_two_qubit_gate(empty_layer):
    empty_layer.add_gates(gates.CX(Qubit(3), Qubit(1)))
    with pytest.raises(
        DuplicateQubitError,
        match=re.escape(
            "For gate MZ(Qubit(3), probability=0.0): "
            "qubits {Qubit(3)} were identified as "
            "being duplicates in the layer."
        ),
    ):
        empty_layer.add_gates(gates.MZ(Qubit(3)))


@pytest.mark.parametrize(
    "gate1",
    [
        gates.H_XY(Qubit(0)),
        gates.CX(Qubit(2), Qubit(0)),
        gates.RX(Qubit(0)),
        gates.ISWAP_DAG(Qubit(0), Qubit(3)),
    ],
)
@pytest.mark.parametrize(
    "gate2",
    [
        gates.I(Qubit(0)),
        gates.ISWAP(Qubit(5), Qubit(0)),
        gates.MPP(
            MeasurementPauliProduct(
                [InvertiblePauliX(Qubit(0)), InvertiblePauliY(Qubit(1))]
            )
        ),
        gates.S(Qubit(0)),
    ],
)
def test_error_is_raised_on_combinations_of_gates_which_both_act_on_qubit_zero(
    empty_layer, gate1, gate2
):
    empty_layer.add_gates(gate1)
    with pytest.raises(
        DuplicateQubitError,
        match=re.escape(
            f"For gate {gate2}: qubits {{Qubit(0)}} "
            "were identified as being duplicates in the "
            "layer."
        ),
    ):
        empty_layer.add_gates(gate2)


class TestGateLayerApproxEquals:
    def test_two_gate_layers_with_the_identical_gates_in_the_same_order_are_approx_equal(
        self,
    ):
        assert GateLayer([gates.X(Qubit(0)), gates.H(Qubit(1))]).approx_equals(
            GateLayer([gates.X(Qubit(0)), gates.H(Qubit(1))])
        )

    def test_two_gate_layers_with_non_measurement_gates_in_different_order_are_approx_equal(
        self,
    ):
        assert GateLayer([gates.X(0), gates.Z(1)]).approx_equals(
            GateLayer([gates.Z(1), gates.X(0)])
        )

    def test_two_gate_layers_with_measurement_gates_in_different_order_are_not_approx_equal(
        self,
    ):
        assert not GateLayer([gates.MX(0), gates.MZ(1)]).approx_equals(
            GateLayer([gates.MZ(1), gates.MX(0)])
        )

    def test_two_gate_layers_with_different_gates_are_not_approx_equal(self):
        assert not GateLayer([gates.X(0), gates.MZ(1)]).approx_equals(
            GateLayer([gates.X(0), gates.MZ(2)])
        )

    def test_two_gate_layers_with_measurement_gates_approx_equal_are_approx_equal_default_tol(
        self,
    ):
        assert GateLayer([gates.X(0), gates.Z(1), gates.MZ(2, 0.001)]).approx_equals(
            GateLayer([gates.Z(1), gates.X(0), gates.MZ(2, 0.001000000001)])
        )

    def test_two_gate_layers_with_measurement_gates_approx_equal_are_approx_equal_other_tol(
        self,
    ):
        assert GateLayer([gates.X(0), gates.Z(1), gates.MZ(2, 0.001)]).approx_equals(
            GateLayer([gates.Z(1), gates.X(0), gates.MZ(2, 0.00100001)]), abs_tol=1e-8
        )

    @pytest.mark.parametrize(("abs_tol", "rel_tol"), [(1e-9, 1e-5), (1e-8, 0.0)])
    def test_two_gate_layers_with_measurement_gates_not_approx_equal_are_not_approx_equal(
        self, abs_tol, rel_tol
    ):
        assert GateLayer([gates.X(0), gates.Z(1), gates.MZ(2, 0.001)]).approx_equals(
            GateLayer([gates.Z(1), gates.X(0), gates.MZ(2, 0.00100001)]),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )

    @pytest.mark.parametrize(
        "other_layer",
        [
            NoiseLayer(PauliXError(0, 0.001)),
            Detector([MeasurementRecord(-1)]),
            Observable(0, [MeasurementRecord(-1)]),
            ShiftCoordinates((0, 0, 1)),
            Circuit(GateLayer(gates.H(0)), iterations=3),
        ],
    )
    def test_gate_layer_and_non_gate_layer_are_not_approx_equal(self, other_layer):
        gate_layer = GateLayer(gates.H(0))
        assert not gate_layer.approx_equals(other_layer)


class TestGateLayerEquality:
    def test_two_gate_layers_with_the_identical_gates_in_the_same_order_are_equal(self):
        assert GateLayer([gates.X(Qubit(0)), gates.H(Qubit(1))]) == GateLayer(
            [gates.X(Qubit(0)), gates.H(Qubit(1))]
        )

    def test_two_gate_layers_with_non_measurement_gates_in_different_order_are_equal(
        self,
    ):
        assert GateLayer([gates.X(0), gates.Z(1)]) == GateLayer(
            [gates.Z(1), gates.X(0)]
        )

    def test_two_gate_layers_with_measurement_gates_in_different_order_are_not_equal(
        self,
    ):
        assert GateLayer([gates.MX(0), gates.MZ(1)]) != GateLayer(
            [gates.MZ(1), gates.MX(0)]
        )

    def test_two_gate_layers_with_different_gates_are_not_equal(self):
        assert GateLayer([gates.X(0), gates.MZ(1)]) != GateLayer(
            [gates.X(0), gates.MZ(2)]
        )


def test_replacing_gate_which_is_not_in_the_gate_layer_does_nothing():
    gate_layer = GateLayer(gates.X(0))
    gate_layer.replace_gates({gates.Z(1): lambda _: gates.Z(1)})
    assert gate_layer == GateLayer(gates.X(0))


@pytest.mark.parametrize(
    ("old_gate", "new_gate"),
    [
        (gates.H(Qubit(1)), gates.X(Qubit(1))),
        (gates.MPP([PauliX(0), PauliY(1)]), gates.MPP([PauliZ(0), PauliY(1)])),
        (gates.RX(Qubit(1)), gates.RY(Qubit(1))),
    ],
)
def test_ability_to_replace_single_gate_object(old_gate, new_gate):
    gate_layer = GateLayer(old_gate)
    gate_layer.replace_gates({old_gate: lambda _: new_gate})
    assert gate_layer == GateLayer(new_gate)


def test_replacing_measurement_gate_places_gate_at_same_position():
    gate_layer = GateLayer([gates.MX(0), gates.MZ(1)])
    gate_layer.replace_gates({gates.MX(0): lambda gate: gates.MY(gate.qubit)})
    assert gate_layer == GateLayer([gates.MY(0), gates.MZ(1)])


def test_ability_to_replace_all_reset_gates_of_a_given_type():
    gate_layer = GateLayer([gates.RX(Qubit(i)) for i in range(5)])
    expected_gate_layer = GateLayer([gates.RZ(Qubit(i)) for i in range(5)])
    gate_layer.replace_gates({gates.RX: lambda gate: gates.RZ(gate.qubit)})
    assert gate_layer == expected_gate_layer


def test_ability_to_replace_all_measurement_gates_of_a_given_type():
    gate_layer = GateLayer([gates.MZ(Qubit(i)) for i in range(5)])
    expected_gate_layer = GateLayer(
        [gates.MZ(Qubit(i), probability=0.001) for i in range(5)]
    )
    gate_layer.replace_gates(
        {gates.MZ: lambda gate: gates.MZ(gate.qubit, probability=0.001)}
    )
    assert gate_layer == expected_gate_layer


@pytest.mark.parametrize(
    "replacement_policy",
    [
        {gates.Z(0): lambda _: gates.Z(1)},
        {gates.MZ(1): lambda _: gates.MZ(2)},
        {gates.Z: lambda _: gates.Z(1)},
        {gates.MZ: lambda _: gates.MZ(2)},
    ],
)
def test_replacing_gates_with_new_gate_which_acts_on_existing_qubit_raises_error(
    replacement_policy,
):
    gate_layer = GateLayer([gates.Z(0), gates.MZ(1), gates.X(2)])
    with pytest.raises(
        DuplicateQubitError,
        match=r"For gate .*: qubits .* "
        "were identified as being duplicates in the "
        "layer.",
    ):
        gate_layer.replace_gates(replacement_policy)


def test_hash(empty_layer, gate_layer):
    assert hash(empty_layer) == hash(empty_layer)
    assert hash(empty_layer) == hash(copy(empty_layer))
    assert hash(empty_layer) == hash(deepcopy(empty_layer))
    assert hash(empty_layer) != hash(gate_layer)


class TestStimCircuit:
    @pytest.mark.parametrize(
        ("gate", "expected_circuit"),
        [
            (gates.X(Qubit(4)), stim.Circuit("X 4")),
            (gates.Y(Qubit(3)), stim.Circuit("Y 3")),
            (gates.RX(Qubit(5)), stim.Circuit("RX 5")),
            (gates.MX(Qubit(2)), stim.Circuit("MX 2")),
            (~gates.MZ(Qubit(3)), stim.Circuit("MZ !3")),
            (gates.MRY(Qubit(3)), stim.Circuit("MRY 3")),
            (gates.MY(Qubit(4), 0.002), stim.Circuit("MY(0.002) 4")),
            (~gates.MRX(Qubit(2), 0.8), stim.Circuit("MRX(0.8) !2")),
            (gates.MY(Qubit(5), 0.0), stim.Circuit("MY 5")),
            (~gates.MRX(Qubit(3), 0.0), stim.Circuit("MRX !3")),
        ],
    )
    def test_stim_circuit_of_layer_with_one_single_qubit_gate_is_expected(
        self, empty_layer: GateLayer, gate, expected_circuit, empty_circuit
    ):
        empty_layer.add_gates(gate)
        empty_layer.permute_stim_circuit(empty_circuit)
        assert empty_circuit == expected_circuit

    @pytest.mark.parametrize(
        ("gate", "expected_circuit"),
        [
            (gates.CX(Qubit(0), Qubit(1)), stim.Circuit("CX 0 1")),
            (gates.ISWAP_DAG(Qubit(4), Qubit(2)), stim.Circuit("ISWAP_DAG 4 2")),
            (gates.CXSWAP(Qubit(0), Qubit(1)), stim.Circuit("CXSWAP 0 1")),
            (gates.ISWAP(Qubit(0), Qubit(1)), stim.Circuit("ISWAP 0 1")),
            (gates.SWAP(Qubit(0), Qubit(1)), stim.Circuit("SWAP 0 1")),
        ],
    )
    def test_stim_circuit_with_layer_that_contains_a_single_two_qubit_gate(
        self, empty_layer: GateLayer, gate, expected_circuit, empty_circuit
    ):
        empty_layer.add_gates(gate)
        empty_layer.permute_stim_circuit(empty_circuit)
        assert empty_circuit == expected_circuit
        # Enable CZSWAP test if Stim > 1.13.0.
        # TODO: If the condition is met, this part is
        # executed for every test parameter. There should
        # be a clause for executing once.
        if CURRENT_STIM_VERSION >= STIM_VERSION_V1_13_0:
            gate = gates.CZSWAP(Qubit(0), Qubit(1))
            expected_circuit = stim.Circuit("CZSWAP 0 1")
            empty_layer = GateLayer()
            empty_circuit = stim.Circuit()
            empty_layer.add_gates(gate)
            empty_layer.permute_stim_circuit(empty_circuit)
            assert empty_circuit == expected_circuit

    @pytest.mark.parametrize(
        ("mpp_gate", "expected_circuit"),
        [
            (gates.MPP(PauliX(Qubit(0))), stim.Circuit("MPP X0")),
            (gates.MPP(InvertiblePauliZ(Qubit(2))), stim.Circuit("MPP Z2")),
            (gates.MPP(~InvertiblePauliY(Qubit(4))), stim.Circuit("MPP !Y4")),
            (
                gates.MPP(
                    MeasurementPauliProduct([PauliX(Qubit(2)), PauliY(Qubit(3))])
                ),
                stim.Circuit("MPP X2*Y3"),
            ),
            (
                gates.MPP(
                    MeasurementPauliProduct(
                        [
                            ~InvertiblePauliZ(Qubit(3)),
                            PauliZ(Qubit(4)),
                            PauliZ(Qubit(5)),
                        ]
                    )
                ),
                stim.Circuit("MPP !Z3*Z4*Z5"),
            ),
            (gates.MPP(PauliX(Qubit(0)), 0.001), stim.Circuit("MPP(0.001) X0")),
            (
                gates.MPP(~InvertiblePauliZ(Qubit(2)), 0.02),
                stim.Circuit("MPP(0.02) !Z2"),
            ),
            (
                gates.MPP(
                    MeasurementPauliProduct([PauliY(Qubit(3)), PauliZ(Qubit(4))]), 0.09
                ),
                stim.Circuit("MPP(0.09) Y3*Z4"),
            ),
            (
                gates.MPP(
                    MeasurementPauliProduct(
                        [
                            PauliY(Qubit(0)),
                            ~InvertiblePauliX(Qubit(1)),
                            PauliY(Qubit(2)),
                        ]
                    ),
                    0.02,
                ),
                stim.Circuit("MPP(0.02) Y0*!X1*Y2"),
            ),
            (
                gates.MPP(
                    MeasurementPauliProduct(
                        [
                            PauliX(Qubit(0)),
                            ~InvertiblePauliY(Qubit(1)),
                            ~InvertiblePauliZ(Qubit(2)),
                        ]
                    ),
                    0.01,
                ),
                stim.Circuit("MPP(0.01) X0*!Y1*!Z2"),
            ),
        ],
    )
    def test_stim_circuit_with_layer_that_contains_a_single_mpp_gate(
        self, empty_layer: GateLayer, mpp_gate, expected_circuit, empty_circuit
    ):
        empty_layer.add_gates(mpp_gate)
        empty_layer.permute_stim_circuit(empty_circuit)
        assert empty_circuit == expected_circuit

    # @pytest.mark.parametrize("gate_class",
    #                          gates.ONE_QUBIT_GATES | gates.RESET_GATES | gates.MEASUREMENT_GATES -
    #                          {gates.MPP}
    #                          )
    # def test_stim_string_on_same_gate_is_on_the_same_line_for_single_qubit_gates(
    #         self, empty_layer: GateLayer, gate_class, empty_circuit):
    #     empty_layer.add_gates([gate_class(Qubit(4)),
    #                            gate_class(Qubit(2))])
    #     empty_layer.permute_stim_circuit(empty_circuit)
    #     assert len(str(empty_circuit).split("\n")) == 1

    @pytest.mark.parametrize("gate_class", gates.ONE_QUBIT_GATES | gates.RESET_GATES)
    def test_stim_string_of_same_gate_is_on_same_line_when_separated_by_other_gate(
        self, empty_circuit, empty_layer: GateLayer, gate_class
    ):
        empty_layer.add_gates(
            [gate_class(Qubit(0)), gates.MX(Qubit(1)), gate_class(Qubit(2))]
        )
        empty_layer.permute_stim_circuit(empty_circuit)
        assert len(str(empty_circuit).split("\n")) == 2

    @pytest.mark.parametrize("gate_class", gates.TWO_QUBIT_GATES)
    def test_stim_string_on_same_gate_is_on_the_same_line_for_two_qubit_gates(
        self, empty_layer: GateLayer, gate_class, empty_circuit
    ) -> None:
        if CURRENT_STIM_VERSION < STIM_VERSION_V1_13_0 and gate_class == gates.CZSWAP:
            pytest.skip(
                "CZSWAP gate has been introduced in Stim v1.13.0. "
                "See https://github.com/quantumlib/Stim/releases/tag/v1.13.0. "
                f"Current Stim version is {CURRENT_STIM_VERSION}."
            )
        empty_layer.add_gates(
            [gate_class(Qubit(0), Qubit(1)), gate_class(Qubit(2), Qubit(3))]
        )
        empty_layer.permute_stim_circuit(empty_circuit)
        assert len(str(empty_circuit).split("\n")) == 1

    # @pytest.mark.parametrize("gate_class", gates.MEASUREMENT_GATES \
    #     - {gates.MPP})
    # def test_circuit_layer_stim_string_with_same_prob_measurement_is_on_one_line(
    #         self, empty_layer: GateLayer, gate_class, empty_circuit):
    #     empty_layer.add_gates([gate_class(Qubit(4), 0.002),
    #                            gate_class(Qubit(1), 0.002)])
    #     empty_layer.permute_stim_circuit(empty_circuit)
    #     assert len(str(empty_circuit).split("\n")) == 1

    @pytest.mark.parametrize(
        ("gate1", "gate2"),
        [
            (gates.MPP(PauliX(Qubit(3))), gates.MPP(PauliX(Qubit(2)))),
            (
                gates.MPP(
                    MeasurementPauliProduct([PauliY(Qubit(2)), PauliZ(Qubit(5))])
                ),
                gates.MPP(
                    MeasurementPauliProduct([PauliZ(Qubit(1)), PauliX(Qubit(10))])
                ),
            ),
            (
                gates.MPP(
                    MeasurementPauliProduct([PauliX(Qubit(3)), PauliZ(Qubit(5))])
                ),
                gates.MPP(PauliY(Qubit(4))),
            ),
            (gates.MPP(PauliZ(Qubit(4)), 0.001), gates.MPP(PauliY(Qubit(2)), 0.001)),
            (
                gates.MPP(
                    MeasurementPauliProduct([PauliZ(Qubit(4)), PauliZ(Qubit(2))]), 0.002
                ),
                gates.MPP(PauliY(Qubit(8)), 0.002),
            ),
        ],
    )
    def test_gate_layer_stim_string_with_same_mpp_gates_is_on_one_line(
        self, empty_layer: GateLayer, gate1, gate2, empty_circuit
    ):
        empty_layer.add_gates([gate1, gate2])
        empty_layer.permute_stim_circuit(empty_circuit)
        assert len(str(empty_circuit).split("\n")) == 1

    def test_gate_layer_measurements_come_out_in_the_same_order_they_went_in(
        self, empty_layer: GateLayer, empty_circuit
    ):
        empty_layer.add_gates([gates.MX(Qubit(0)), gates.MZ(Qubit(1))])
        empty_layer.permute_stim_circuit(empty_circuit)
        assert empty_circuit == stim.Circuit("MX 0\nMZ 1")

    @pytest.mark.parametrize(
        ("gate", "qubit_mapping", "expected_stim_circuit"),
        [
            (gates.X((0, 0)), {Qubit((0, 0)): 0}, stim.Circuit("X 0")),
            (gates.Y((0, 0)), {Qubit((0, 0)): 3}, stim.Circuit("Y 3")),
            (
                gates.SWAP("qubit1", "qubit2"),
                {Qubit("qubit1"): 1, Qubit("qubit2"): 2},
                stim.Circuit("SWAP 1 2"),
            ),
        ],
    )
    def test_gate_layer_gives_expected_stim_circuit_when_giving_a_qubit_mapping(
        self,
        empty_layer: GateLayer,
        gate,
        qubit_mapping,
        expected_stim_circuit,
        empty_circuit,
    ):
        empty_layer.add_gates(gate)
        empty_layer.permute_stim_circuit(empty_circuit, qubit_mapping)
        assert empty_circuit == expected_stim_circuit


class TestQubitTransforms:
    @pytest.mark.parametrize(
        "gates", [(gates.X(0), gates.Y(1)), (gates.MX(0), gates.MZ(1))]
    )
    def test_error_is_raised_if_transforming_qubits_leads_to_multiple_operations_on_same_qubit(
        self, empty_layer: GateLayer, gates
    ):
        empty_layer.add_gates(gates)
        with pytest.raises(
            DuplicateQubitError,
            match=r"For gate .*: qubits {Qubit\(1\)} "
            "were identified as being duplicates in the "
            "layer.",
        ):
            empty_layer.transform_qubits({0: 1})

    @pytest.mark.parametrize(
        "gates", [(gates.X(0), gates.Y(1)), (gates.MX(0), gates.MZ(1))]
    )
    def test_error_is_raised_if_mapping_is_not_one_to_one(
        self, empty_layer: GateLayer, gates
    ):
        empty_layer.add_gates(gates)
        with pytest.raises(
            DuplicateQubitError,
            match=r"For gate .*: qubits {Qubit\(2\)} "
            "were identified as being duplicates in the "
            "layer.",
        ):
            empty_layer.transform_qubits({0: 2, 1: 2})

    @pytest.mark.parametrize(
        "gates",
        [
            (gates.X(0), gates.CX(1, 2)),
            (gates.SQRT_X(0), gates.MX(1), gates.MZ(2)),
            (gates.MPP([PauliX(0), PauliY(1), PauliZ(2)])),
        ],
    )
    def test_transforming_qubits_changes_qubits_in_mapping(
        self, empty_layer: GateLayer, gates
    ):
        empty_layer.add_gates(gates)
        empty_layer.transform_qubits({0: 3, 1: 4, 2: 5})
        assert empty_layer.qubits == frozenset((Qubit(3), Qubit(4), Qubit(5)))

    @pytest.mark.parametrize(
        ("input_layer", "id_mapping", "expected_layer"),
        [
            (
                GateLayer([gates.X(0), gates.CX(1, 2)]),
                {0: 3, 1: 4, 2: 5},
                GateLayer([gates.X(3), gates.CX(4, 5)]),
            ),
            (
                GateLayer([gates.MX(0), gates.CZ(1, 2)]),
                {0: 3, 1: 4, 2: 5},
                GateLayer([gates.MX(3), gates.CZ(4, 5)]),
            ),
            (
                GateLayer(gates.MPP([PauliX(0), InvertiblePauliY(1), PauliZ(2)])),
                {0: 3, 1: 4, 2: 5},
                GateLayer(gates.MPP([PauliX(3), InvertiblePauliY(4), PauliZ(5)])),
            ),
        ],
    )
    def test_transforming_qubits_changes_layer_for_qubits_in_mapping(
        self, input_layer: GateLayer, id_mapping, expected_layer: GateLayer
    ):
        input_layer.transform_qubits(id_mapping)
        assert input_layer == expected_layer
