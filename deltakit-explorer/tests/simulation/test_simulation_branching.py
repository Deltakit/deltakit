# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import pytest

from deltakit_explorer import simulation
from deltakit_explorer._api._client import Client
from deltakit_explorer._utils._decorators import (
    _split_into_batches,
    validate_and_split_simulation,
)
from deltakit_explorer.data._data_analysis import has_leakage
from deltakit_explorer.types import LeakageFlags, Measurements


@pytest.mark.parametrize(
    ("circuit", "leakage"),
    [
        (
            """
            QUBIT_COORDS(0, 12) 140
            QUBIT_COORDS(0, 21) 141
            QUBIT_COORDS(2, 18) 142
            QUBIT_COORDS(1, 22) 143
            R 0 1 4 3 2 6 7 5 8
            TICK
            DEPOLARIZE1(0.002) 0 1 4 3
            H 0 2 3 5 7 8 9 10 11 12 17
            TICK
            DEPOLARIZE1(0.002) 0 2 3 5
            """,
            False,
        ),
        (
            """
            TICK
            X_ERROR(0.02) 0 1 3 4 2
            LEAKAGE(0.01) 0 1 3 4 2
            S 0 3 1
            TICK
            DEPOLARIZE1(0.001) 0 3 1
            RELAX(0.002) 0 3 1 2 4
            """,
            True,
        ),
        (
            """
            X_ERROR(0.0408163) 2 4
            HERALD_LEAKAGE_EVENT(0.1) 2 4
            M(0.01) 2 4
            TICK
            DEPOLARIZE1(0.02) 0 3 1
            RELAX(0.04) 0 3 1
            """,
            True,
        ),
    ],
)
def test_leakage_detection_is_correct(circuit, leakage):
    assert has_leakage(circuit) == leakage


def test_leakage_simulation_calls_server_v1(mocker):
    circuit = """
    H 0 1
    LEAKAGE(0.01) 0
    M 0 1
    """
    client = Client.get_instance(api_version=1)
    mocker.patch.object(client._api, "_get_query", return_value="query")
    mocker_execute = mocker.patch.object(
        client._api, "execute_query",
        return_value={"simulateWithStim": [{"uid": "duck://3232"}]},
    )
    mmts, _ = client.simulate_stim_circuit(circuit, 100)
    assert mmts.as_numpy().shape == (1, 2)
    mocker_execute.assert_called_once()


@pytest.mark.parametrize(
    ("total", "batch", "result"),
    [
        (500, 100, [100, 100, 100, 100, 100]),
        (501, 100, [100, 100, 100, 100, 100, 1]),
        (501, 101, [101, 101, 101, 101, 97]),
        (100, 101, [100]),
        (101, 101, [101]),
        (1, 1, [1]),
        (1, 10000000, [1]),
    ],
)
def test_batch_splitting_works_correctly(total, batch, result):
    assert _split_into_batches(total, batch) == result


def test_simulator_decorator_splits_on_server(mocker):
    circuit = "H 0 1\nHERALD_LEAKAGE_EVENT 0\nM 0 1"
    client = Client.get_instance()

    mmts_batch = Measurements([[0, 0]] * 100_000)  # one batch
    leak_batch = LeakageFlags([[0]] * 100_000)  # one batch

    # pytest-mock replaces the method together with the decorator,
    mock = mocker.Mock(return_value=(mmts_batch, leak_batch))
    # so we re-apply the decorator to reproduce original behaviour
    decorated_mock = validate_and_split_simulation(mock)
    mocker.patch.object(Client, "simulate_stim_circuit", decorated_mock)
    # this will trigger mock method 1.7M/100K=17 times
    mmts, leak = simulation.simulate_with_stim(circuit, 1_700_000, client=client)
    # splits into 17 batches
    assert mock.call_count == 17
    # and successfully joins them
    assert mmts.as_numpy().shape == (1_700_000, 2)
    assert leak.as_numpy().shape == (1_700_000, 1)
