# (c) Copyright Riverlane 2020-2025.
import pytest
from deltakit_circuit.gates import (CX, CY, CZ, Gate, ISWAP, MEASUREMENT_GATES,
                                    MY, ONE_QUBIT_GATES, RESET_GATES, RX, RY, RZ,
                                    SQRT_X, SQRT_XX, TWO_QUBIT_GATES, H, I, X,
                                    Y, Z)
from deltakit_explorer.qpu._native_gate_set import (ExhaustiveGateSet,
                                                    NativeGateSet,
                                                    NativeGateSetAndTimes)


class TestNativeGateSetAndTimes:
    def test_1q_gateset_output(self):
        assert NativeGateSetAndTimes(
            one_qubit_gates={X: 0.1, Y: 0.2, Z: 0.3, I: 0.4}
        ).one_qubit_gates == {X: 0.1, Y: 0.2, Z: 0.3, I: 0.4}

    def test_1q_gateset_when_gate_not_stim_gate(self):
        input_gates = {X: 1.0, Y: 1.0, "FGFG": 1.0, I: 1.0}
        with pytest.raises(
            ValueError, match=r".* one-qubit gate list is not a valid gate.*"
        ):
            NativeGateSetAndTimes(one_qubit_gates=input_gates)

    def test_1q_gateset_when_time_negative_float(self):
        with pytest.raises(
            ValueError,
            match="A gate time must be a non-negative float but that for X is -1.0.",
        ):
            NativeGateSetAndTimes(one_qubit_gates={X: -1.0})

    def test_2q_gateset_output(self):
        assert NativeGateSetAndTimes(
            two_qubit_gates={CX: 0.1, CY: 0.2, CZ: 0.3}
        ).two_qubit_gates == {CX: 0.1, CY: 0.2, CZ: 0.3}

    def test_2q_gateset_when_gate_not_stim_gate(self):
        input_gates = {CX: 0.1, CY: 0.2, "FGFG": 0.3}
        with pytest.raises(
            ValueError, match=r".* two-qubit gate list is not a valid gate.*"
        ):
            NativeGateSetAndTimes(two_qubit_gates=input_gates)

    def test_2q_gateset_when_time_negative_float(self):
        with pytest.raises(
            ValueError,
            match="A gate time must be a non-negative float but that for CY is -1.0.",
        ):
            NativeGateSetAndTimes(two_qubit_gates={CY: -1.0})

    def test_reset_gateset_output(self):
        assert NativeGateSetAndTimes(
            reset_gates={RX: 0.1, RY: 0.2, RZ: 0.3}
        ).reset_gates == {RX: 0.1, RY: 0.2, RZ: 0.3}

    def test_reset_gateset_when_gate_not_stim_gate(self):
        input_gates = {RX: 0.1, "FGFG": 0.2}
        with pytest.raises(
            ValueError, match=r".* reset gate list is not a valid gate.*"
        ):
            NativeGateSetAndTimes(reset_gates=input_gates)

    def test_reset_gateset_when_time_negative_float(self):
        with pytest.raises(
            ValueError,
            match="A gate time must be a non-negative float but that for RX is -1.0.",
        ):
            NativeGateSetAndTimes(reset_gates={RX: -1.0})

    def test_measurement_gateset_when_gate_not_stim_gate(self):
        input_gates = {MY: 0.1, "FGFG": 0.2}
        with pytest.raises(
            ValueError, match=r".* measurement gate list is not a valid gate.*"
        ):
            NativeGateSetAndTimes(measurement_gates=input_gates)

    @pytest.mark.parametrize(
        "gate, time, gate_type_attr",
        [
            (SQRT_X, 2.0, "one_qubit_gates"),
            (SQRT_XX, 0.001, "two_qubit_gates"),
            (RZ, 5.6, "reset_gates"),
            (MY, 3.14, "measurement_gates"),
        ],
    )
    def test_add_gate(self, gate, time, gate_type_attr):
        native_gate_set = NativeGateSetAndTimes()
        native_gate_set.add_gate(gate, time)
        assert gate in native_gate_set.native_gates

        gate_set = getattr(native_gate_set, gate_type_attr)
        assert gate_set[gate] == time

    def test_add_invalid_gate(self):
        class MY_GATE(Gate):
            stim_string = "MY_GATE"
        native_gate_set = NativeGateSetAndTimes()
        with pytest.raises(ValueError, match=r"Unknown gate..."):
            native_gate_set.add_gate(MY_GATE, 2.71)

    @pytest.mark.parametrize(
        "gate, gate_type_attr",
        [
            (SQRT_X, "one_qubit_gates"),
            (SQRT_XX, "two_qubit_gates"),
            (RZ, "reset_gates"),
        ],
    )
    def test_add_gate_with_default_time(self, gate, gate_type_attr):
        native_gate_set = NativeGateSetAndTimes()
        native_gate_set.add_gate(gate)
        assert gate in native_gate_set.native_gates

        gate_set = getattr(native_gate_set, gate_type_attr)
        assert gate_set[gate] == 1.0

    @pytest.mark.parametrize("native_gates", [
        None,
        NativeGateSet([H], [ISWAP], [RY], [MY]),
    ])
    def test_from_times(self, native_gates):
        time_1_qubit_gate = 25e-9
        time_2_qubit_gate = 34e-9
        time_measurement = 500e-9
        time_reset = 160e-9

        native_gates_times = NativeGateSetAndTimes.from_times(
            time_1_qubit_gate, time_2_qubit_gate, time_reset, time_measurement, native_gates=native_gates,
        )

        if native_gates is None:
            native_gates = ExhaustiveGateSet()

        # The times of the returned native gate set are as specified
        assert set(native_gates_times.one_qubit_gates.values()) == {time_1_qubit_gate}
        assert set(native_gates_times.two_qubit_gates.values()) == {time_2_qubit_gate}
        assert set(native_gates_times.measurement_gates.values()) == {time_measurement}
        assert set(native_gates_times.reset_gates.values()) == {time_reset}

        # The returned native gate set has the same gates as the provided/exhaustive gate
        assert set(native_gates_times.one_qubit_gates.keys()) == set(native_gates.one_qubit_gates)
        assert set(native_gates_times.two_qubit_gates.keys()) == set(native_gates.two_qubit_gates)
        assert set(native_gates_times.measurement_gates.keys()) == set(native_gates.measurement_gates)
        assert set(native_gates_times.reset_gates.keys()) == set(native_gates.reset_gates)


class TestNativeGateSet:
    def test_1q_gateset_output(self):
        assert NativeGateSet(one_qubit_gates={X, Y, Z, I}).one_qubit_gates == {
            X: 1.0,
            Y: 1.0,
            Z: 1.0,
            I: 1.0,
        }

    def test_2q_gateset_output(self):
        assert NativeGateSet(two_qubit_gates={CX, CY, CZ}).two_qubit_gates == {
            CX: 1.0,
            CY: 1.0,
            CZ: 1.0,
        }

    def test_reset_gateset_output(self):
        assert NativeGateSet(reset_gates={RX, RY, RZ}).reset_gates == {
            RX: 1.0,
            RY: 1.0,
            RZ: 1.0,
        }


class TestExhaustiveGateSet:
    @pytest.fixture(scope="class")
    def default_exhaustive_gateset(request):
        return ExhaustiveGateSet()

    def test_1q_gateset_output(self, default_exhaustive_gateset):
        assert default_exhaustive_gateset.one_qubit_gates == dict.fromkeys(ONE_QUBIT_GATES, 1)

    def test_2q_gateset_output(self, default_exhaustive_gateset):
        assert default_exhaustive_gateset.two_qubit_gates == dict.fromkeys(TWO_QUBIT_GATES, 1)

    def test_reset_gateset_output(self, default_exhaustive_gateset):
        assert default_exhaustive_gateset.reset_gates == dict.fromkeys(RESET_GATES, 1)

    def test_measurement_gateset_output(self, default_exhaustive_gateset):
        assert default_exhaustive_gateset.measurement_gates == dict.fromkeys(MEASUREMENT_GATES, 1)
