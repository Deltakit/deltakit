# (c) Copyright Riverlane 2020-2025.
from itertools import permutations

import pytest
import stim

from deltakit_circuit import (
    Circuit,
    Detector,
    GateLayer,
    MeasurementRecord,
    NoiseLayer,
    Observable,
    PauliProduct,
    PauliX,
    PauliY,
    PauliZ,
    Qubit,
    ShiftCoordinates,
    noise_channels,
)
from deltakit_circuit.gates import H


@pytest.fixture
def empty_layer() -> NoiseLayer:
    return NoiseLayer()


@pytest.mark.parametrize(
    "noise_channel",
    [
        noise_channels.PauliXError(Qubit(0), 0.1),
        noise_channels.PauliYError(Qubit(1), 0.2),
        noise_channels.PauliZError(Qubit(4), 0.1),
        noise_channels.PauliChannel1(Qubit(0), 0.2, 0.1, 0.3),
        noise_channels.PauliChannel2(Qubit(0), Qubit(1), 0.1),
        noise_channels.Depolarise1(Qubit(0), 0.3),
        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.12),
        noise_channels.CorrelatedError(
            PauliProduct([PauliX(Qubit(0)), PauliY(Qubit(1))]), 0.2
        ),
        noise_channels.ElseCorrelatedError(
            PauliProduct([PauliZ(Qubit(0)), PauliY(Qubit(1))]), 0.1
        ),
    ],
)
def test_adding_a_noise_channel_to_a_noise_layer_adds_it(
    empty_layer: NoiseLayer, noise_channel
):
    empty_layer.add_noise_channels(noise_channel)
    assert noise_channel in empty_layer.noise_channels


@pytest.mark.parametrize(
    ("noise_channels", "expected_repr"),
    [
        (
            noise_channels.Depolarise1(Qubit(2), 0.01),
            "NoiseLayer([\n    DEPOLARIZE1(Qubit(2), probability=0.01)\n])",
        ),
        (
            [
                noise_channels.PauliXError(Qubit(0), 0.01),
                noise_channels.Depolarise2(Qubit(1), Qubit(2), 0.02),
            ],
            "NoiseLayer([\n"
            "    X_ERROR(Qubit(0), probability=0.01)\n"
            "    DEPOLARIZE2(qubit1=Qubit(1), qubit2=Qubit(2), probability=0.02)\n"
            "])",
        ),
    ],
)
def test_repr_of_noise_layer_returns_expected_representation(
    empty_layer: NoiseLayer, noise_channels, expected_repr
):
    empty_layer.add_noise_channels(noise_channels)
    assert repr(empty_layer) == expected_repr


class TestNoiseLayerEquality:
    @pytest.mark.parametrize(
        "noise_channels",
        [
            (noise_channels.PauliXError(Qubit(0), 0.1),),
            (
                noise_channels.Depolarise1(Qubit(1), 0.2),
                noise_channels.PauliChannel1(Qubit(0), 0.1, 0.2, 0.3),
            ),
            (
                noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.2),
                noise_channels.PauliZError(Qubit(2), 0.2),
            ),
        ],
    )
    def test_noise_layers_with_the_same_noise_channels_are_equal(self, noise_channels):
        assert NoiseLayer(noise_channels) == NoiseLayer(noise_channels)

    def test_noise_layer_with_uncorrelated_errors_in_different_order_are_equal(self):
        uncorrelated_noise = [
            noise_channels.PauliXError(Qubit(0), 0.02),
            noise_channels.Depolarise1(Qubit(0), 0.01),
        ]
        assert NoiseLayer(uncorrelated_noise) == NoiseLayer(
            reversed(uncorrelated_noise)
        )

    def test_noise_layers_with_different_number_of_uncorrelated_errors_are_not_equal(
        self,
    ):
        noise_layer1 = NoiseLayer(
            [
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.Depolarise1(Qubit(1), 0.001),
            ]
        )
        noise_layer2 = NoiseLayer(
            [
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.Depolarise1(Qubit(1), 0.02),
            ]
        )
        assert noise_layer1 != noise_layer2

    def test_noise_layers_with_same_order_of_correlated_errors_are_equal(self):
        noise_layer1 = NoiseLayer(
            [
                noise_channels.CorrelatedError(PauliX(Qubit(0)), 0.01),
                noise_channels.ElseCorrelatedError(PauliX(Qubit(1)), 0.02),
            ]
        )
        noise_layer2 = NoiseLayer(
            [
                noise_channels.CorrelatedError(PauliX(Qubit(0)), 0.01),
                noise_channels.ElseCorrelatedError(PauliX(Qubit(1)), 0.02),
            ]
        )
        assert noise_layer1 == noise_layer2

    def test_noise_layers_with_different_order_of_correlated_errors_are_not_equal(self):
        noise_layer1 = NoiseLayer(
            [
                noise_channels.CorrelatedError(PauliX(Qubit(0)), 0.01),
                noise_channels.ElseCorrelatedError(PauliX(Qubit(1)), 0.02),
                noise_channels.CorrelatedError(PauliZ(Qubit(1)), 0.01),
                noise_channels.ElseCorrelatedError(PauliZ(Qubit(0)), 0.01),
            ]
        )
        noise_layer2 = NoiseLayer(
            [
                noise_channels.CorrelatedError(PauliZ(Qubit(1)), 0.01),
                noise_channels.ElseCorrelatedError(PauliZ(Qubit(0)), 0.01),
                noise_channels.CorrelatedError(PauliX(Qubit(0)), 0.01),
                noise_channels.ElseCorrelatedError(PauliX(Qubit(1)), 0.02),
            ]
        )
        assert noise_layer1 != noise_layer2

    @pytest.mark.parametrize(
        ("noise_channels1", "noise_channels2"),
        permutations(
            [
                (noise_channels.PauliXError(Qubit(0), 0.1),),
                (
                    noise_channels.Depolarise1(Qubit(1), 0.2),
                    noise_channels.PauliChannel1(Qubit(0), 0.1, 0.2, 0.3),
                ),
                (
                    noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.2),
                    noise_channels.PauliZError(Qubit(2), 0.2),
                ),
            ],
            2,
        ),
    )
    def test_noise_layers_with_different_noise_channels_are_different(
        self, noise_channels1, noise_channels2
    ):
        assert NoiseLayer(noise_channels1) != NoiseLayer(noise_channels2)

    def test_two_noise_layers_which_should_be_equal_are_equal(self):
        noise_layer1 = NoiseLayer(
            [
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                noise_channels.PauliZError(Qubit(2), 0.002),
                noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
            ]
        )
        noise_layer2 = NoiseLayer(
            [
                noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                noise_channels.PauliZError(Qubit(2), 0.002),
                noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
            ]
        )
        assert noise_layer1 == noise_layer2


class TestNoiseLayerApproxEquals:
    @pytest.mark.parametrize(
        "noise_channels",
        [
            (noise_channels.PauliXError(Qubit(0), 0.1),),
            (
                noise_channels.Depolarise1(Qubit(1), 0.2),
                noise_channels.PauliChannel1(Qubit(0), 0.1, 0.2, 0.3),
            ),
            (
                noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.2),
                noise_channels.PauliZError(Qubit(2), 0.2),
            ),
        ],
    )
    def test_noise_layers_with_the_same_noise_channels_are_approx_equal(
        self, noise_channels
    ):
        assert NoiseLayer(noise_channels).approx_equals(NoiseLayer(noise_channels))

    def test_noise_layer_with_uncorrelated_errors_in_different_order_are_approx_equal(
        self,
    ):
        uncorrelated_noise = [
            noise_channels.PauliXError(Qubit(0), 0.02),
            noise_channels.Depolarise1(Qubit(0), 0.01),
        ]
        assert NoiseLayer(uncorrelated_noise).approx_equals(
            NoiseLayer(reversed(uncorrelated_noise))
        )

    def test_noise_layers_with_different_number_of_uncorrelated_errors_are_not_approx_equal(
        self,
    ):
        noise_layer1 = NoiseLayer(
            [
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.Depolarise1(Qubit(1), 0.001),
            ]
        )
        noise_layer2 = NoiseLayer(
            [
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.Depolarise1(Qubit(1), 0.02),
            ]
        )
        assert not noise_layer1.approx_equals(noise_layer2)

    def test_noise_layers_with_same_order_of_correlated_errors_are_approx_equal(self):
        noise_layer1 = NoiseLayer(
            [
                noise_channels.CorrelatedError(PauliX(Qubit(0)), 0.01),
                noise_channels.ElseCorrelatedError(PauliX(Qubit(1)), 0.02),
            ]
        )
        noise_layer2 = NoiseLayer(
            [
                noise_channels.CorrelatedError(PauliX(Qubit(0)), 0.01),
                noise_channels.ElseCorrelatedError(PauliX(Qubit(1)), 0.02),
            ]
        )
        assert noise_layer1.approx_equals(noise_layer2)

    def test_noise_layers_with_different_order_of_correlated_errors_are_not_approx_equal(
        self,
    ):
        noise_layer1 = NoiseLayer(
            [
                noise_channels.CorrelatedError(PauliX(Qubit(0)), 0.01),
                noise_channels.ElseCorrelatedError(PauliX(Qubit(1)), 0.02),
                noise_channels.CorrelatedError(PauliZ(Qubit(1)), 0.01),
                noise_channels.ElseCorrelatedError(PauliZ(Qubit(0)), 0.01),
            ]
        )
        noise_layer2 = NoiseLayer(
            [
                noise_channels.CorrelatedError(PauliZ(Qubit(1)), 0.01),
                noise_channels.ElseCorrelatedError(PauliZ(Qubit(0)), 0.01),
                noise_channels.CorrelatedError(PauliX(Qubit(0)), 0.01),
                noise_channels.ElseCorrelatedError(PauliX(Qubit(1)), 0.02),
            ]
        )
        assert not noise_layer1.approx_equals(noise_layer2)

    @pytest.mark.parametrize(
        ("noise_channels1", "noise_channels2"),
        permutations(
            [
                (noise_channels.PauliXError(Qubit(0), 0.1),),
                (
                    noise_channels.Depolarise1(Qubit(1), 0.2),
                    noise_channels.PauliChannel1(Qubit(0), 0.1, 0.2, 0.3),
                ),
                (
                    noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.2),
                    noise_channels.PauliZError(Qubit(2), 0.2),
                ),
            ],
            2,
        ),
    )
    def test_noise_layers_with_different_noise_channels_are_not_approx_equal(
        self, noise_channels1, noise_channels2
    ):
        assert not NoiseLayer(noise_channels1).approx_equals(
            NoiseLayer(noise_channels2)
        )

    def test_two_noise_layers_which_should_be_equal_are_approx_equal(self):
        noise_layer1 = NoiseLayer(
            [
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                noise_channels.PauliZError(Qubit(2), 0.002),
                noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
            ]
        )
        noise_layer2 = NoiseLayer(
            [
                noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                noise_channels.PauliZError(Qubit(2), 0.002),
                noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
            ]
        )
        assert noise_layer1.approx_equals(noise_layer2)

    @pytest.mark.parametrize(
        ("noise_layer1", "noise_layer2"),
        [
            (
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.02),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                        noise_channels.PauliZError(Qubit(2), 0.002),
                        noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02000000001),
                        noise_channels.PauliXError(Qubit(0), 0.02000000001),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01000000001),
                        noise_channels.PauliZError(Qubit(2), 0.002000000001),
                        noise_channels.ElseCorrelatedError(
                            PauliZ(Qubit(2)), 0.03000000001
                        ),
                    ]
                ),
            ),
            (
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.02),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                        noise_channels.PauliXError(Qubit(0), 0.002),
                        noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02000000001),
                        noise_channels.PauliXError(Qubit(0), 0.002000000001),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01000000001),
                        noise_channels.PauliXError(Qubit(0), 0.02000000001),
                        noise_channels.ElseCorrelatedError(
                            PauliZ(Qubit(2)), 0.03000000001
                        ),
                    ]
                ),
            ),
        ],
    )
    def test_two_noise_layers_with_approx_equal_probs_are_approx_equal_default_tol(
        self, noise_layer1, noise_layer2
    ):
        assert noise_layer1.approx_equals(noise_layer2)

    @pytest.mark.parametrize(
        ("noise_layer1", "noise_layer2", "rel_tol", "abs_tol"),
        [
            (
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.02),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                        noise_channels.PauliZError(Qubit(2), 0.002),
                        noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.020000001),
                        noise_channels.PauliXError(Qubit(0), 0.020000001),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.010000001),
                        noise_channels.PauliZError(Qubit(2), 0.0020000001),
                        noise_channels.ElseCorrelatedError(
                            PauliZ(Qubit(2)), 0.030000001
                        ),
                    ]
                ),
                1e-7,
                0.0,
            ),
            (
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.02),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                        noise_channels.PauliZError(Qubit(2), 0.002),
                        noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.020000001),
                        noise_channels.PauliXError(Qubit(0), 0.020000001),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.010000001),
                        noise_channels.PauliZError(Qubit(2), 0.002000001),
                        noise_channels.ElseCorrelatedError(
                            PauliZ(Qubit(2)), 0.030000001
                        ),
                    ]
                ),
                1e-9,
                1e-9,
            ),
            (
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.0202),
                        noise_channels.PauliXError(Qubit(0), 0.02),
                        noise_channels.PauliXError(Qubit(0), 0.0204),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.0202),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.02),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.0204),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.0203),
                        noise_channels.PauliXError(Qubit(0), 0.0199),
                        noise_channels.PauliXError(Qubit(0), 0.0201),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.0203),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.0199),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.0201),
                    ]
                ),
                1e-2,
                0.0,
            ),
            (
                NoiseLayer(
                    [
                        noise_channels.PauliChannel1(0, 0.1, 0.1, 0.1),
                        noise_channels.PauliChannel1(0, 0.1, 0.2, 0.2),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.PauliChannel1(0, 0.1, 0.1, 0.1),
                        noise_channels.PauliChannel1(0, 0.0999999999999, 0.2, 0.2),
                    ]
                ),
                1e-2,
                0.0,
            ),
        ],
    )
    def test_two_noise_layers_with_approx_equal_probs_are_approx_equal_other_tol(
        self, noise_layer1, noise_layer2, rel_tol, abs_tol
    ):
        assert noise_layer1.approx_equals(
            noise_layer2, rel_tol=rel_tol, abs_tol=abs_tol
        )

    @pytest.mark.parametrize(
        ("noise_layer1", "noise_layer2"),
        [
            (
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.02),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                        noise_channels.PauliZError(Qubit(2), 0.002),
                        noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.0200000001),
                        noise_channels.PauliXError(Qubit(0), 0.0200000001),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.0100000001),
                        noise_channels.PauliZError(Qubit(2), 0.00200000001),
                        noise_channels.ElseCorrelatedError(
                            PauliZ(Qubit(2)), 0.0300000001
                        ),
                    ]
                ),
            ),
            (
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.02),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                        noise_channels.PauliXError(Qubit(0), 0.002),
                        noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.0200000001),
                        noise_channels.PauliXError(Qubit(0), 0.00200000001),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.0100000001),
                        noise_channels.PauliXError(Qubit(0), 0.0200000001),
                        noise_channels.ElseCorrelatedError(
                            PauliZ(Qubit(2)), 0.0300000001
                        ),
                    ]
                ),
            ),
        ],
    )
    def test_two_noise_layers_with_not_approx_equal_probs_are_not_approx_equal_default_tol(
        self, noise_layer1, noise_layer2
    ):
        assert not noise_layer1.approx_equals(noise_layer2)

    @pytest.mark.parametrize(
        ("noise_layer1", "noise_layer2", "rel_tol", "abs_tol"),
        [
            (
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.02),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                        noise_channels.PauliZError(Qubit(2), 0.002),
                        noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.020000001),
                        noise_channels.PauliXError(Qubit(0), 0.020000001),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.010000001),
                        noise_channels.PauliZError(Qubit(2), 0.0020000001),
                        noise_channels.ElseCorrelatedError(
                            PauliZ(Qubit(2)), 0.030000001
                        ),
                    ]
                ),
                1e-8,
                0.0,
            ),
            (
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.02),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                        noise_channels.PauliZError(Qubit(2), 0.002),
                        noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.020000001),
                        noise_channels.PauliXError(Qubit(0), 0.020000001),
                        noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.010000001),
                        noise_channels.PauliZError(Qubit(2), 0.002000001),
                        noise_channels.ElseCorrelatedError(
                            PauliZ(Qubit(2)), 0.030000001
                        ),
                    ]
                ),
                1e-9,
                1e-10,
            ),
            (
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.0202),
                        noise_channels.PauliXError(Qubit(0), 0.02),
                        noise_channels.PauliXError(Qubit(0), 0.0204),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.0202),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.02),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.0204),
                    ]
                ),
                NoiseLayer(
                    [
                        noise_channels.PauliXError(Qubit(0), 0.0203),
                        noise_channels.PauliXError(Qubit(0), 0.0199),
                        noise_channels.PauliXError(Qubit(0), 0.0201),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.0203),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.0199),
                        noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.0201),
                    ]
                ),
                1e-3,
                0.0,
            ),
        ],
    )
    def test_two_noise_layers_with_not_approx_equal_probs_are_not_approx_equal_other_tol(
        self, noise_layer1, noise_layer2, rel_tol, abs_tol
    ):
        assert not noise_layer1.approx_equals(
            noise_layer2, rel_tol=rel_tol, abs_tol=abs_tol
        )

    @pytest.mark.parametrize(
        "other_layer",
        [
            GateLayer(H(0)),
            Detector([MeasurementRecord(-1)]),
            Observable(0, [MeasurementRecord(-1)]),
            ShiftCoordinates((0, 0, 1)),
            Circuit(GateLayer(H(0)), iterations=3),
        ],
    )
    def test_noise_layer_and_non_noise_layer_are_not_approx_equal(self, other_layer):
        noise_layer = NoiseLayer(
            [
                noise_channels.PauliXError(Qubit(0), 0.02),
                noise_channels.CorrelatedError(PauliX(Qubit(1)), 0.01),
                noise_channels.PauliZError(Qubit(2), 0.002),
                noise_channels.ElseCorrelatedError(PauliZ(Qubit(2)), 0.03),
                noise_channels.Depolarise2(Qubit(3), Qubit(4), 0.02),
            ]
        )
        assert not noise_layer.approx_equals(other_layer)


@pytest.mark.parametrize(
    "noise_channels",
    [
        (noise_channels.PauliXError(Qubit(i), 0.1) for i in range(4)),
        [
            noise_channels.Depolarise2(Qubit(i), Qubit(i + 1), 0.2)
            for i in range(0, 5, 2)
        ],
    ],
)
def test_adding_a_multiple_noise_channels_adds_them_all(
    empty_layer: NoiseLayer, noise_channels
):
    empty_layer.add_noise_channels(noise_channels)
    assert all(
        noise_channel in empty_layer.noise_channels for noise_channel in noise_channels
    )


@pytest.mark.parametrize(
    "noise_channels",
    [
        (noise_channels.PauliYError(Qubit(0), 0.2),),
        (noise_channels.Depolarise1(Qubit(i), 0.01) for i in range(2)),
    ],
)
def test_noise_layer_can_be_initialised_with_noise_channels(noise_channels):
    layer = NoiseLayer(noise_channels)
    assert all(
        noise_channel in layer.noise_channels for noise_channel in noise_channels
    )


@pytest.mark.parametrize(
    ("noise_channel_class", "probabilities"),
    [
        (noise_channels.PauliXError, (0.1,)),
        (noise_channels.PauliYError, (0.1,)),
        (noise_channels.PauliZError, (0.1,)),
        (noise_channels.Depolarise1, (0.1,)),
        (noise_channels.PauliChannel1, (0.1, 0.2, 0.3)),
    ],
)
def test_adding_single_qubit_noise_on_same_qubit_adds_both_noises(
    empty_layer: NoiseLayer, noise_channel_class, probabilities
):
    empty_layer.add_noise_channels(
        [
            noise_channel_class(Qubit(0), *probabilities),
            noise_channel_class(Qubit(0), *probabilities),
        ]
    )
    assert len(empty_layer.noise_channels) == 2


def test_adding_depolarise2_noise_on_same_qubit_adds_both_noises(
    empty_layer: NoiseLayer,
):
    empty_layer.add_noise_channels(
        [
            noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.1),
            noise_channels.Depolarise2(Qubit(0), Qubit(1), 0.1),
        ]
    )
    assert len(empty_layer.noise_channels) == 2


def test_adding_pauli_channel2_noise_on_same_qubit_adds_both_noises(
    empty_layer: NoiseLayer,
):
    empty_layer.add_noise_channels(
        [
            noise_channels.PauliChannel2(Qubit(0), Qubit(1), 0.1),
            noise_channels.PauliChannel2(Qubit(0), Qubit(1), 0.1),
        ]
    )
    assert len(empty_layer.noise_channels) == 2


@pytest.mark.parametrize(
    "noise_channel_class",
    [noise_channels.CorrelatedError, noise_channels.ElseCorrelatedError],
)
def test_adding_correlated_noise_on_the_same_channel_adds_both_noises(
    empty_layer: NoiseLayer, noise_channel_class
):
    pauli_product = PauliProduct([PauliX(Qubit(0)), PauliX(Qubit(1))])
    empty_layer.add_noise_channels(
        [
            noise_channel_class(pauli_product, 0.1),
            noise_channel_class(pauli_product, 0.1),
        ]
    )
    assert len(empty_layer.noise_channels) == 2


class TestStimCircuit:
    @pytest.mark.parametrize(
        ("noise_channel", "expected_circuit"),
        [
            (noise_channels.PauliXError(Qubit(0), 0.2), stim.Circuit("X_ERROR(0.2) 0")),
            (noise_channels.PauliYError(Qubit(2), 0.1), stim.Circuit("Y_ERROR(0.1) 2")),
            (noise_channels.PauliZError(Qubit(1), 0.4), stim.Circuit("Z_ERROR(0.4) 1")),
            (
                noise_channels.PauliChannel1(Qubit(3), 0.1, 0.2, 0.3),
                stim.Circuit("PAULI_CHANNEL_1(0.1, 0.2, 0.3) 3"),
            ),
            (
                noise_channels.PauliChannel2(
                    Qubit(1),
                    Qubit(0),
                    0.02,
                    0.01,
                    0.04,
                    0.03,
                    0.013,
                    0.045,
                    0.056,
                    0.021,
                    0.052,
                    0.014,
                    0.031,
                    0.025,
                    0.042,
                    0.041,
                    0.04,
                ),
                stim.Circuit(
                    "PAULI_CHANNEL_2(0.02, 0.01, 0.04, 0.03, 0.013, "
                    "0.045, 0.056, 0.021, 0.052, 0.014, 0.031, 0.025, 0.042, 0.041, "
                    "0.04) 1 0"
                ),
            ),
            (
                noise_channels.Depolarise1(Qubit(9), 0.4),
                stim.Circuit("DEPOLARIZE1(0.4) 9"),
            ),
            (
                noise_channels.Depolarise2(Qubit(2), Qubit(3), 0.5),
                stim.Circuit("DEPOLARIZE2(0.5) 2 3"),
            ),
            (
                noise_channels.CorrelatedError(
                    PauliProduct([PauliX(Qubit(3)), PauliY(Qubit(2))]), 0.2
                ),
                stim.Circuit("CORRELATED_ERROR(0.2) X3 Y2"),
            ),
            (
                noise_channels.ElseCorrelatedError(
                    PauliProduct([PauliZ(Qubit(2)), PauliZ(Qubit(3))]), 0.25
                ),
                stim.Circuit("ELSE_CORRELATED_ERROR(0.25) Z2 Z3"),
            ),
        ],
    )
    def test_layer_with_single_noise_channel_equals_expected_circuit(
        self, empty_layer: NoiseLayer, noise_channel, expected_circuit, empty_circuit
    ):
        empty_layer.add_noise_channels(noise_channel)
        empty_layer.permute_stim_circuit(empty_circuit)
        assert empty_circuit == expected_circuit

    @pytest.mark.parametrize(
        "noise_channel_class",
        [
            noise_channels.PauliXError,
            noise_channels.PauliYError,
            noise_channels.PauliZError,
            noise_channels.Depolarise1,
        ],
    )
    def test_same_single_qubit_noise_channels_are_on_same_stim_circuit_line(
        self, empty_layer: NoiseLayer, noise_channel_class, empty_circuit
    ):
        empty_layer.add_noise_channels(noise_channel_class(Qubit(0), 0.1))
        empty_layer.add_noise_channels(noise_channel_class(Qubit(1), 0.1))
        empty_layer.permute_stim_circuit(empty_circuit)
        assert len(str(empty_circuit).split("\n")) == 1

    def test_two_uncorrelated_noise_channels_that_are_separated_get_merged_to_same_line(
        self, empty_layer: NoiseLayer, empty_circuit
    ):
        empty_layer.add_noise_channels(
            [
                noise_channels.PauliXError(0, 0.1),
                noise_channels.PauliYError(1, 0.2),
                noise_channels.PauliXError(2, 0.1),
            ]
        )
        empty_layer.permute_stim_circuit(empty_circuit)
        assert empty_circuit == stim.Circuit("X_ERROR(0.1) 0 2\nY_ERROR(0.2) 1")

    def test_same_pauli_channel_1_noise_is_on_the_same_stim_circuit_line(
        self, empty_layer: NoiseLayer, empty_circuit
    ):
        probabilities = (0.1, 0.2, 0.3)
        empty_layer.add_noise_channels(
            noise_channels.PauliChannel1(Qubit(0), *probabilities)
        )
        empty_layer.add_noise_channels(
            noise_channels.PauliChannel1(Qubit(1), *probabilities)
        )
        empty_layer.permute_stim_circuit(empty_circuit)
        assert len(str(empty_circuit).split("\n")) == 1

    def test_same_pauli_channel_2_noise_is_on_the_same_stim_circuit_line(
        self, empty_layer: NoiseLayer, empty_circuit
    ):
        probabilities = (
            0.01,
            0.011,
            0.012,
            0.013,
            0.014,
            0.015,
            0.016,
            0.017,
            0.018,
            0.019,
            0.02,
            0.021,
            0.022,
            0.023,
            0.024,
        )
        empty_layer.add_noise_channels(
            noise_channels.PauliChannel2(Qubit(0), Qubit(1), *probabilities)
        )
        empty_layer.add_noise_channels(
            noise_channels.PauliChannel2(Qubit(2), Qubit(3), *probabilities)
        )
        empty_layer.permute_stim_circuit(empty_circuit)
        assert len(str(empty_circuit).split("\n")) == 1

    @pytest.mark.parametrize(
        ("noise_channel", "qubit_mapping", "expected_stim_circuit"),
        [
            (
                noise_channels.Depolarise1((0, 1), 0.2),
                {Qubit((0, 1)): 0},
                stim.Circuit("DEPOLARIZE1(0.2) 0"),
            ),
            (
                noise_channels.PauliChannel1(4, 0.01, 0.0, 0.0),
                {Qubit(4): 2},
                stim.Circuit("PAULI_CHANNEL_1(0.01, 0.0, 0.0) 2"),
            ),
        ],
    )
    def test_noise_channel_with_qubit_mapping_matches_expected_stim_circuit(
        self,
        empty_layer: NoiseLayer,
        noise_channel,
        qubit_mapping,
        expected_stim_circuit,
        empty_circuit,
    ):
        empty_layer.add_noise_channels(noise_channel)
        empty_layer.permute_stim_circuit(empty_circuit, qubit_mapping)
        assert empty_circuit == expected_stim_circuit

    def test_correlated_errors_retain_their_order_in_stim(
        self, empty_layer: NoiseLayer, empty_circuit
    ):
        empty_layer.add_noise_channels(
            [
                noise_channels.CorrelatedError(PauliX(0), 0.01),
                noise_channels.ElseCorrelatedError(PauliZ(0), 0.01),
                noise_channels.CorrelatedError(PauliX(1), 0.01),
                noise_channels.ElseCorrelatedError(PauliZ(1), 0.01),
            ]
        )
        empty_layer.permute_stim_circuit(empty_circuit)
        assert empty_circuit == stim.Circuit(
            "CORRELATED_ERROR(0.01) X0\n"
            "ELSE_CORRELATED_ERROR(0.01) Z0\n"
            "CORRELATED_ERROR(0.01) X1\n"
            "ELSE_CORRELATED_ERROR(0.01) Z1"
        )


class TestQubitTransforms:
    @pytest.fixture
    def noisy_layer(self, empty_layer: NoiseLayer) -> NoiseLayer:
        empty_layer.add_noise_channels(
            [
                noise_channels.PauliXError(0, 0.01),
                noise_channels.CorrelatedError([PauliX(1), PauliY(2)], 0.01),
                noise_channels.Depolarise1(Qubit(3), 0.01),
                noise_channels.PauliChannel1(Qubit(4), 0.01, 0.02, 0.03),
            ]
        )
        return empty_layer

    def test_transforming_qubits_not_in_id_mapping_does_not_change_noise_channels(
        self, noisy_layer: NoiseLayer
    ):
        old_noise_channels = noisy_layer.noise_channels
        noisy_layer.transform_qubits({})
        assert noisy_layer.noise_channels == old_noise_channels

    def test_transforming_qubits_in_a_noise_layer_changes_noise_channels(
        self, noisy_layer: NoiseLayer
    ):
        noisy_layer.transform_qubits(
            {
                qubit_id.unique_identifier: -qubit_id.unique_identifier
                for qubit_id in noisy_layer.qubits
            }
        )
        assert noisy_layer.noise_channels == (
            noise_channels.PauliXError(0, 0.01),
            noise_channels.Depolarise1(Qubit(-3), 0.01),
            noise_channels.PauliChannel1(Qubit(-4), 0.01, 0.02, 0.03),
            noise_channels.CorrelatedError([PauliX(-1), PauliY(-2)], 0.01),
        )

    def test_transforming_qubits_in_a_noise_layer_changes_qubits(
        self, noisy_layer: NoiseLayer
    ):
        noisy_layer.transform_qubits(
            {
                qubit_id.unique_identifier: -qubit_id.unique_identifier
                for qubit_id in noisy_layer.qubits
            }
        )
        assert noisy_layer.qubits == frozenset(
            [Qubit(0), Qubit(-1), Qubit(-2), Qubit(-3), Qubit(-4)]
        )
