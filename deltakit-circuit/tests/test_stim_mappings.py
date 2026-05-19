# (c) Copyright Riverlane 2020-2025.
from deltakit_circuit.gates import GATE_MAPPING
from deltakit_circuit.noise_channels import NOISE_CHANNEL_MAPPING

ALL_DELTAKIT_STIM_GATE_STRINGS = (
    "I",
    "X",
    "Y",
    "Z",
    "C_XYZ",
    "C_ZYX",
    "H",
    "H_XY",
    "H_XZ",
    "H_YZ",
    "S",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "SQRT_Z",
    "SQRT_Z_DAG",
    "S_DAG",
    "CNOT",
    "CX",
    "CXSWAP",
    "CY",
    "CZ",
    "CZSWAP",
    "ISWAP",
    "ISWAP_DAG",
    "SQRT_XX",
    "SQRT_XX_DAG",
    "SQRT_YY",
    "SQRT_YY_DAG",
    "SQRT_ZZ",
    "SQRT_ZZ_DAG",
    "SWAP",
    "XCX",
    "XCY",
    "XCZ",
    "YCX",
    "YCY",
    "YCZ",
    "ZCX",
    "ZCY",
    "ZCZ",
    "M",
    "MPP",
    "MR",
    "MRX",
    "MRY",
    "MRZ",
    "MX",
    "MY",
    "MZ",
    "R",
    "RX",
    "RY",
    "RZ",
)

ALL_DELTAKIT_STIM_NOISE_CHANNEL_STRINGS = (
    "CORRELATED_ERROR",
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "E",
    "ELSE_CORRELATED_ERROR",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
)


def test_every_valid_deltakit_stim_gate_can_be_mapped_to_a_deltakit_circuit_gate():
    assert all(
        deltakit_stim_string in GATE_MAPPING
        for deltakit_stim_string in ALL_DELTAKIT_STIM_GATE_STRINGS
    )


def test_every_valid_deltakit_stim_noise_channel_can_be_mapped_to_a_deltakit_circuit_noise_channel():
    assert all(
        deltakit_stim_string in NOISE_CHANNEL_MAPPING
        for deltakit_stim_string in ALL_DELTAKIT_STIM_NOISE_CHANNEL_STRINGS
    )
