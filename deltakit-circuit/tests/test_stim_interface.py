# (c) Copyright Riverlane 2020-2025.
import deltakit_stim
import pytest


@pytest.mark.parametrize(
    ("deltakit_stim_circuit1", "deltakit_stim_circuit2"),
    [
        (
            deltakit_stim.Circuit("I 0\nM 0\nDETECTOR rec[-1]"),
            deltakit_stim.Circuit("X_ERROR(0) 0\nM 0\nDETECTOR rec[-1]"),
        ),
        (
            deltakit_stim.Circuit("I 0\nMZ(1) 0\nDETECTOR rec[-1]"),
            deltakit_stim.Circuit("X_ERROR(1) 0\nMZ 0\nDETECTOR rec[-1]"),
        ),
        (
            deltakit_stim.Circuit("X 0\nM 0\nDETECTOR rec[-1]"),
            deltakit_stim.Circuit("X 0\nM(0) 0\nDETECTOR rec[-1]"),
        ),
        (
            deltakit_stim.Circuit("X 0\nM !0\nDETECTOR rec[-1]"),
            deltakit_stim.Circuit("X 0\nM(0) 0\nDETECTOR rec[-1]"),
        ),
        (
            deltakit_stim.Circuit("X 0\nMZ(1) 0\nDETECTOR rec[-1]"),
            deltakit_stim.Circuit("X_ERROR(1) 0\nMZ(0) 0\nDETECTOR rec[-1]"),
        ),
    ],
)
def test_deltakit_stim_detector_error_model_is_equal_when_error_probability_is_equal(
    deltakit_stim_circuit1, deltakit_stim_circuit2
):
    assert (
        deltakit_stim_circuit1.detector_error_model()
        == deltakit_stim_circuit2.detector_error_model()
    )
