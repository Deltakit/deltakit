# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from collections import namedtuple
from math import exp, log

import pytest
from deltakit_explorer.analysis import \
    predict_distance_for_quops, predict_quops_at_distance


Parameters = namedtuple("Parameters", ["p_0", "lambda_"])

def alternative_lep_per_round(p_0: float, lambda_: float, d: int) -> float:
    return p_0 * exp(-log(lambda_) * (d + 1) / 2)

@pytest.fixture
def default_parameters():
    return Parameters(1e-7, 8)


@pytest.mark.parametrize("distance", ([3, 11, 19]))
def test_predict_quops_at_distance_method(
    default_parameters: Parameters,
    distance: int
):
    expected_lep_per_round = alternative_lep_per_round(
        default_parameters.p_0, default_parameters.lambda_, distance
    )
    expected_lep = 0.5 * (1 - pow(1 - 2 * expected_lep_per_round, distance))
    prediction = predict_quops_at_distance(*default_parameters, distance)
    assert pytest.approx(1 / expected_lep) == prediction


def test_predict_distance_for_quops_method_when_quops_too_small(
    default_parameters: Parameters,
):
    with pytest.raises(ValueError, match="Number of QuOps should be at least 2"):
        predict_distance_for_quops(*default_parameters, 1)

@pytest.mark.parametrize("lambda0", [-1, 0, 0.9999, 1.000])
def test_predict_distance_for_quops_method_when_lambda_too_small(
    lambda0,
):
    with pytest.raises(ValueError, match="Lambda should be greater than 0"):
        predict_distance_for_quops(0.001, lambda0, 10)


def test_predict_distance_for_quops_method_raises_when_quops_too_big():
    # Create a system that has a lambda very close to threshold and try and reach
    # a large number of QuOps with this.
    with pytest.raises(
        ValueError,
        match="Could not find a solution"
    ):
        predict_distance_for_quops(4e-2, 1.05, 1e9)


@pytest.mark.parametrize(
    "p0, lambda_, distance, quops",
    [
        (1.0e-3, 2.0, 11, 5819.090965902487),
        (1.0e-2, 1.5, 5, 68.30475477033615),
        (1.0e-4, 2.05, 13, 117040.75112970827),
        # megaqoup
        (1.0e-4, 2.1, 21, 1667989.0518755822),
    ]
)
def test_predict_qoups_by_distance_constants(p0, lambda_, distance, quops):
    prediction = predict_quops_at_distance(p0, lambda_, distance)
    assert pytest.approx(prediction, rel=1e-4) == quops


@pytest.mark.parametrize(
    "p0, lambda_, distance, quops",
    [
        (1.0e-3, 2.0, 11, 5800),
        # TODO: this test fails, as it actually finds
        # a solution better: no QEC (d=1).
        # in this case the function is not descending
        # Question: how do I treat this?
        # (1.0e-2, 1.5, 5, 65),
        (1.0e-4, 2.05, 13, 110000),
        # megaqoup
        (1.0e-4, 2.1, 21, 1660000),
    ]
)
def test_predict_distance_for_qoups_constants(p0, lambda_, distance, quops):
    dist = predict_distance_for_quops(p0, lambda_, quops)
    assert pytest.approx(dist, rel=1e-4) == distance
