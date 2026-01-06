# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import numpy as np
import pytest

from deltakit_explorer.enums import DataFormat
from deltakit_explorer.types import BinaryDataType, DataString, DecodingResult


@pytest.mark.parametrize("pred", [None, {"uid": None}, {"uid": "duck://2020"}])
def test_constructor_from_json(pred):
    json = {
        "fails": 15,
        "shots": 1000,
        "times": [0.4, 0.3, 12.4],
        "counts": [333, 333, 334],
        "predictionsFile": pred,
    }
    res = DecodingResult(**json)
    assert sum(res.counts) == res.shots


@pytest.mark.parametrize(
    ("fails", "shots", "prob", "dev"),
    [
        (13, 65, 0.2, 0.04961389383568338),
        (0, 100000, 0, 0),
        (50000, 100000, 0.5, 0.00158113883),
    ],
)
def test_error_rate(fails, shots, prob, dev):
    result = DecodingResult(
        fails=fails,
        shots=shots,
        times=[],
        counts=[],
    )
    assert pytest.approx(result.get_logical_error_probability()) == prob
    assert pytest.approx(result.get_standard_deviation()) == dev


def test_get_predictions_none():
    assert (
        DecodingResult(
            fails=10,
            shots=10,
            times=[],
            counts=[],
            predictionsFile=None,
        ).predictions
        is None
    )


def test_get_predictions_none_uid():
    assert (
        DecodingResult(
            fails=10,
            shots=10,
            times=[],
            counts=[],
            predictionsFile={"uid": None},
        ).predictions
        is None
    )


def test_get_predictions_good_data():
    assert (
        DecodingResult(
            fails=10,
            shots=10,
            times=[],
            counts=[],
            predictions_format=DataFormat.F01,
            predictionsFile={"uid": str(DataString("0101\n1010"))},
        ).predictions.as_01_string()
        == "0101\n1010"
    )


def test_get_predictions_fail_to_parse():
    with pytest.raises(ValueError):
        DecodingResult(
            fails=10,
            shots=10,
            times=[],
            counts=[],
            predictions_format=DataFormat.F01,
            predictionsFile={"uid": "duck://0201"},
        ).predictions.as_numpy()


@pytest.mark.parametrize("items", [1, 2, 3])
def test_join_works_as_predicted(items):
    data = [[0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]]

    result = []
    for i in range(items):
        predictions = BinaryDataType(data[i : i + 1], DataFormat.B8)
        result.append(
            DecodingResult(
                fails=0,
                shots=1,
                times=[0.0],
                counts=[1],
            )
        )
        result[-1].predictions = predictions
    joined = DecodingResult.combine(result)
    assert np.allclose(joined.predictions.data.content, data[:items])


@pytest.mark.parametrize("items", [2, 3])
def test_join_raises_on_nonaligned_rows(items):
    data = [[0, 1, 0, 1, 0], [0, 1, 1], [1, 0, 1, 0]]

    result = []
    for i in range(items):
        predictions = BinaryDataType(data[i : i + 1], DataFormat.B8)
        result.append(
            DecodingResult(
                fails=0,
                shots=1,
                times=[0.0],
                counts=[1],
            )
        )
        result[-1].predictions = predictions
    msg = "all the input array dimensions except for the concatenation axis must match exactly"
    with pytest.raises(ValueError, match=msg):
        DecodingResult.combine(result)


def test_decoding_result_eq():
    res1 = DecodingResult(
        0, 1, [0., 1., 2.], 2,
        predictions_format=DataFormat.F01,
        predictionsFile={"uid": "duck://30300a30310a"}
    )
    res2 = DecodingResult(
        0, 1, [0., 3., 7.], 2,
        predictions_format=DataFormat.CSV,
        predictionsFile={"uid": "duck://302c300a302c310a"}
    )
    assert res1 == res2
