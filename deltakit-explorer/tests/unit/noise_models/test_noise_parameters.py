# (c) Copyright Riverlane 2020-2025.
import numpy as np
import pytest
from deltakit_circuit import (
    Circuit,
    GateLayer,
    NoiseContext,
    Qubit,
    before_measure_flip_probability,
    gates,
    measurement_noise_profile,
)
from deltakit_circuit.gates import CX, X, Y, Z
from deltakit_circuit.noise_channels import Depolarise1, PauliChannel1

from deltakit_explorer.qpu._noise import NoiseParameters
from deltakit_explorer.qpu._noise._noise_parameters import _idle_noise_from_t1_t2


class TestNoiseParameters:
    @pytest.mark.parametrize(
        ("gate_dict", "idle", "reset", "m_noise", "m_flip", "before_gate"),
        [
            (
                {
                    X: lambda gate: Depolarise1(gate.qubit, 0.1),
                    Y: lambda gate: Depolarise1(gate.qubit, 0.1),
                    Z: lambda gate: Depolarise1(gate.qubit, 1.0),
                    CX: lambda gate: Depolarise1(gate.qubit, 0.1),
                },
                lambda qubit: Depolarise1(qubit, 0.1),
                lambda gate: Depolarise1(gate.qubit, 0.5),
                lambda gate: Depolarise1(gate.qubit, 1.0),
                measurement_noise_profile(0.3),
                before_measure_flip_probability(0.1),
            ),
            (
                {
                    X: lambda gate: Depolarise1(gate.qubit, 0.1),
                    Y: lambda gate: Depolarise1(gate.qubit, 0.1),
                    Z: lambda gate: Depolarise1(gate.qubit, 1.0),
                    CX: lambda gate: Depolarise1(gate.qubit, 0.1),
                },
                lambda qubit, t: Depolarise1(qubit, 0.1 * np.exp(-t)),
                lambda gate: Depolarise1(gate.qubit, 0.5),
                lambda gate: Depolarise1(gate.qubit, 1.0),
                measurement_noise_profile(0.3),
                lambda gate: Depolarise1(gate.qubit, 0.5),
            ),
        ],
    )
    def test_pass_for_fixed_values(
        self, gate_dict, idle, reset, m_noise, m_flip, before_gate
    ):
        noise = NoiseParameters(
            gate_noise=gate_dict,
            idle_noise=idle,
            reset_noise=reset,
            measurement_noise_after=m_noise,
            measurement_flip=m_flip,
            measurement_noise_before=before_gate,
        )
        assert noise.gate_noise == gate_dict
        assert noise.idle_noise == idle
        assert noise.reset_noise == reset
        assert noise.measurement_noise_after == m_noise
        assert noise.measurement_flip == m_flip
        assert noise.measurement_noise_before == before_gate

    def test_position_args_raise_error(self):
        with pytest.raises(TypeError, match="positional argument"):
            NoiseParameters(0.01)


class TestIdleNoiseFromT1T2:

    def test_invalid_t1_t2(self):
        t1 = 1
        t2 = 3 * t1

        message = "Relaxation time `t1` must be positive."
        with pytest.raises(ValueError, match=message):
            _idle_noise_from_t1_t2(-t1, t2)

        message = "Dephasing time `t2` must be positive."
        with pytest.raises(ValueError, match=message):
            _idle_noise_from_t1_t2(t1, -t2)

        message = "Dephasing time `t2` must be less than twice relaxation time `t1`."
        with pytest.raises(ValueError, match=message):
            _idle_noise_from_t1_t2(t1, t2)

    def test_typical_case(self):
        # From `Ghosh et al. <https://arxiv.org/abs/1210.5799>`_:
        #
        # "II.B: The asymmetric depolarisation channel (ADC) is such a model,
        # where a decoherent qubit is assumed to suffer from discrete Pauli X
        # (bit-flip) errors, Z (phase flip) errors, or Y (both)..."
        #
        # The probabilities are given in equation (10):
        # p_X = p_Y = (1 - exp(-t/T_1)) / 4
        # p_Z = (1 - \exp(-t/T_2)) / 2 - (1 - \exp(-t/T_1)) / 4
        circuit = Circuit([GateLayer([gates.I(Qubit(0))])])

        t1 = 0.5
        t2 = 0.25
        t = 0.125
        noise_context = NoiseContext(circuit, circuit.layers[0])
        noise_channel_result = _idle_noise_from_t1_t2(t1, t2)(noise_context, t)

        assert isinstance(noise_channel_result, PauliChannel1)
        p_X = p_Y = (1 - np.exp(-t/t1)) / 4
        p_Z = (1 - np.exp(-t/t2)) / 2 - (1 - np.exp(-t/t1)) / 4
        np.testing.assert_allclose(noise_channel_result.probabilities, [p_X, p_Y, p_Z])

    def test_t1_equals_t2(self):
        circuit = Circuit([GateLayer([gates.I(Qubit(0))])])

        t = 1
        noise_context = NoiseContext(circuit, circuit.layers[0])
        noise_channel_result = _idle_noise_from_t1_t2(t, t)(noise_context, t)

        # "If T2 = T1, the ADC reduces to the symmetric depolarisation channel."
        # In this case, $p_X = p_Y = p_Z = (1 - \exp(-t/t1)) / 4$.
        # `Depolarise1` applies a randomly chosen Pauli with a given probability $p / 3$,
        # so the appropriate $p$ to use is $3 * (1 - \exp(-t/t1)) / 4 = 0.75 * (1 - \exp(-t/t1)$
        assert isinstance(noise_channel_result, Depolarise1)
        np.testing.assert_allclose(noise_channel_result.probabilities, 0.75 * (1.0 - np.exp(-t/t)))
