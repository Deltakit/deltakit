import pytest
from deltakit_circuit import (
    Circuit,
    GateLayer,
    NoiseContext,
    Qubit,
    gates,
    noise_channels,
)

from deltakit_explorer.qpu import PhysicalNoise
from deltakit_explorer.qpu._noise._noise_parameters import _idle_noise_from_t1_t2


@pytest.fixture
def noise_params():
    return {
        't1': 20e-6,
        't2': 30e-6,
        'p_1_qubit_gate_error': 0.001,
        'p_2_qubit_gate_error': 0.02,
        'p_reset_error': 0.03,
        'p_meas_qubit_error': 0.04,
        'p_readout_flip': 0.05,
    }


class TestPhysicalNoise:
    def test_physical_noise(self, noise_params):
        # Generate an example circuit with gates of all types
        qubits = [Qubit(0), Qubit(1)]
        circuit = Circuit([GateLayer([gates.I(qubits[0])]),
                           GateLayer([gates.H(qubits[0])]),
                           GateLayer([gates.ISWAP(*qubits)]),
                           GateLayer([gates.RX(qubits[0])]),
                           GateLayer([gates.MX(qubits[0])]),
                           GateLayer([gates.MRZ(qubits[0])]),
                           ])

        noise_parameters = PhysicalNoise(**noise_params)

        # All measurement flip gates are present, and all associated probabilities are `p_meas_qubit_error`
        assert set(noise_parameters.measurement_flip.keys()) == gates.MEASUREMENT_GATES - {gates.HERALD_LEAKAGE_EVENT}
        for key, val in noise_parameters.measurement_flip.items():
            if key != gates.MPP:
                assert val(key(qubits[0])).probability == noise_params['p_readout_flip']

        # Noise channels associated with all gates are depolarising noise with specified error probability
        noise_context = NoiseContext(circuit, circuit.layers[0])
        assert not noise_parameters.gate_noise[0](noise_context)  # I gates are not included in `gate_noise`

        noise_context = NoiseContext(circuit, circuit.layers[1])
        noise_channel = noise_parameters.gate_noise[0](noise_context)[0]
        assert isinstance(noise_channel, noise_channels.Depolarise1)
        assert noise_channel.probability == noise_params['p_1_qubit_gate_error']

        noise_context = NoiseContext(circuit, circuit.layers[2])
        noise_channel = noise_parameters.gate_noise[0](noise_context)[0]
        assert isinstance(noise_channel, noise_channels.Depolarise2)
        assert noise_channel.probability == noise_params['p_2_qubit_gate_error']

        noise_context = NoiseContext(circuit, circuit.layers[3])
        noise_channel = noise_parameters.reset_noise[0](noise_context)[0]
        assert isinstance(noise_channel, noise_channels.Depolarise1)
        assert noise_channel.probability == noise_params['p_reset_error']

        noise_context = NoiseContext(circuit, circuit.layers[4])
        noise_channel = noise_parameters.measurement_noise_after[0](noise_context)[0]
        assert isinstance(noise_channel, noise_channels.Depolarise1)
        assert noise_channel.probability == noise_params['p_meas_qubit_error']
        assert not noise_parameters.measurement_noise_before  # no additional noise before measurements

        noise_context = NoiseContext(circuit, circuit.layers[5])
        noise_channel = noise_parameters.measurement_noise_after[0](noise_context)[0]
        assert isinstance(noise_channel, noise_channels.Depolarise1)
        assert noise_channel.probability == noise_params['p_meas_qubit_error']
        assert not noise_parameters.reset_noise[0](noise_context)  # MRZ is treated as measurement, not reset

        # Check idle noise against _idle_noise_from_t1_t2
        t = 1e-6
        noise_context = NoiseContext(circuit, circuit.layers[0])
        noise_channel_result = noise_parameters.idle_noise(noise_context, t)
        noise_channel_reference = _idle_noise_from_t1_t2(noise_params['t1'], noise_params['t2'])(noise_context, t)
        assert isinstance(noise_channel_result, noise_channels.PauliChannel1)
        assert noise_channel_result.probabilities == noise_channel_reference.probabilities

    def test_position_args_raise_error(self):
        with pytest.raises(TypeError, match="positional argument"):
            PhysicalNoise(0.01)

