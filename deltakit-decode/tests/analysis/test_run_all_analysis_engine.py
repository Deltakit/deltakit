# (c) Copyright Riverlane 2020-2025.
import pathlib
from functools import partial
from typing import List, Protocol
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from deltakit_decode.analysis._decoder_manager import DecoderManager
from deltakit_decode.analysis._run_all_analysis_engine import \
    RunAllAnalysisEngine


@pytest.fixture(params=[0, 1, 4])
def num_decoder_managers(request) -> int:
    return request.param


class MockRunBatchShots(Protocol):
    def __call__(self, batch_size: int, decoder_manager: DecoderManager) -> None: ...


def mock_all_fails_run_batch_shots(batch_size: int, decoder_manager: DecoderManager) -> None:
    distr = decoder_manager.empirical_decoding_error_distribution
    distr.shots += batch_size  # type: ignore[misc]
    distr.fails += batch_size  # type: ignore[misc]
    distr.fails_per_logical = np.asarray([distr.fails])  # type: ignore[misc]


def mock_half_fails_run_batch_shots(batch_size: int, decoder_manager: DecoderManager) -> None:
    "Mock run batch shots which fails exactly half the time"
    distr = decoder_manager.empirical_decoding_error_distribution
    distr.shots += batch_size  # type: ignore[misc]
    assert batch_size % 2 == 0
    distr.fails += batch_size//2  # type: ignore[misc]
    distr.fails_per_logical = np.asarray([distr.fails])  # type: ignore[misc]


mock_all_fails_run_batch_shots_typed: MockRunBatchShots = mock_all_fails_run_batch_shots
mock_half_fails_run_batch_shots_typed: MockRunBatchShots = mock_half_fails_run_batch_shots


@pytest.fixture
def all_fail_decoder_managers(mocker, num_decoder_managers) -> List[DecoderManager]:
    return mock_decoder_managers(mocker, num_decoder_managers, mock_all_fails_run_batch_shots_typed)


@pytest.fixture
def half_fail_decoder_managers(mocker, num_decoder_managers) -> List[DecoderManager]:
    return mock_decoder_managers(mocker, num_decoder_managers, mock_half_fails_run_batch_shots_typed)


def mock_decoder_managers(mocker,
                          num_decoder_managers,
                          mock_run_batch_shots: MockRunBatchShots,
                          ) -> List[DecoderManager]:
    mock_decoder_managers: List[DecoderManager] = []
    for _ in range(num_decoder_managers):
        get_reporter_results: MagicMock = mocker.MagicMock(return_value={
            "test_field_name": "test_field_value"
        })
        mock_decoder_manager: MagicMock = mocker.MagicMock()

        mock_decoding_distribution: MagicMock = mocker.MagicMock()
        mock_decoding_distribution.shots = 0
        mock_decoding_distribution.fails = 0
        mock_decoding_distribution.fails_per_logical = [0]

        type(mock_decoder_manager).empirical_decoding_error_distribution = mock_decoding_distribution
        mock_decoder_manager.run_batch_shots.side_effect = partial(
            mock_run_batch_shots, decoder_manager=mock_decoder_manager)
        mock_decoder_manager.get_reporter_results = get_reporter_results

        mock_decoder_managers.append(mock_decoder_manager)

    return mock_decoder_managers


def test_error_raised_for_invalid_data_directory(all_fail_decoder_managers,
                                                 tmp_path):
    with pytest.raises(NotADirectoryError):
        RunAllAnalysisEngine("test_experiment",
                             max_shots=100,
                             decoder_managers=all_fail_decoder_managers,
                             data_directory=tmp_path / pathlib.Path(str(hash(tmp_path))),
                             )


def test_run_returns_data_frame(all_fail_decoder_managers):
    engine = RunAllAnalysisEngine(
        "test_experiment",
        max_shots=100,
        decoder_managers=all_fail_decoder_managers,
        num_parallel_processes=1
    )
    assert isinstance(engine.run(), pd.DataFrame)


def test_run_outputs_to_file_when_data_directory_is_given(all_fail_decoder_managers, tmp_path):
    expr_name = "test_experiment"
    engine = RunAllAnalysisEngine(
        expr_name,
        max_shots=100,
        decoder_managers=all_fail_decoder_managers,
        num_parallel_processes=1,
        data_directory=tmp_path,
    )
    engine.run()
    pathlib.Path.exists(tmp_path / expr_name)


def test_loop_until_failures_runs_correct_number_of_shots(all_fail_decoder_managers, random_generator):
    fail_num = random_generator.integers(10, 100)
    engine = RunAllAnalysisEngine(
        "test_experiment",
        loop_condition=RunAllAnalysisEngine.loop_until_failures(fail_num),
        decoder_managers=all_fail_decoder_managers,
        num_parallel_processes=1,
        batch_size=1,
        max_shots=10*fail_num
    )
    engine.run()
    for decoder_manager in all_fail_decoder_managers:
        assert decoder_manager.run_batch_shots.call_count == fail_num


def test_max_shots_correct_number_of_shots(all_fail_decoder_managers, random_generator):
    shot_num = random_generator.integers(10, 100)
    engine = RunAllAnalysisEngine(
        "test_experiment",
        decoder_managers=all_fail_decoder_managers,
        num_parallel_processes=1,
        batch_size=1,
        max_shots=shot_num
    )
    engine.run()
    for decoder_manager in all_fail_decoder_managers:
        assert decoder_manager.run_batch_shots.call_count == shot_num


def test_max_shots_param_runs_correct_number_of_shots(all_fail_decoder_managers, random_generator):
    shot_num = random_generator.integers(10, 100)
    batch_size = random_generator.integers(1, 100)

    engine = RunAllAnalysisEngine(
        "test_experiment",
        decoder_managers=all_fail_decoder_managers,
        num_parallel_processes=1,
        batch_size=batch_size,
        max_shots=shot_num
    )
    engine.run()
    for decoder_manager in all_fail_decoder_managers:
        assert decoder_manager.empirical_decoding_error_distribution.shots == shot_num


def test_loop_until_observable_rse_below_threshold_runs_correct_number_of_shots(all_fail_decoder_managers, random_generator):
    "Test when rse target is set to 1.0 engine halts once min_fails fails reached. "
    shot_num = random_generator.integers(10, 100)
    engine = RunAllAnalysisEngine(
        "test_experiment",
        loop_condition=RunAllAnalysisEngine.loop_until_observable_rse_below_threshold(
            1.0, shot_num),
        decoder_managers=all_fail_decoder_managers,
        num_parallel_processes=1,
        batch_size=1,
        max_shots=shot_num
    )
    engine.run()
    for decoder_manager in all_fail_decoder_managers:
        assert decoder_manager.run_batch_shots.call_count == shot_num


def test_loop_until_observable_rse_below_threshold_stops_below_threshold(half_fail_decoder_managers):
    rse_target = 10**np.random.default_rng().uniform(low=-2, high=0)
    engine = RunAllAnalysisEngine(
        "test_experiment",
        loop_condition=RunAllAnalysisEngine.loop_until_observable_rse_below_threshold(
            rse_target, 100),
        decoder_managers=half_fail_decoder_managers,
        num_parallel_processes=1,
        batch_size=2,
        max_shots=1000000
    )
    engine.run()

    for decoder_manager in half_fail_decoder_managers:
        distr = decoder_manager.empirical_decoding_error_distribution
        assert len(distr.fails_per_logical) > 0

        for log_fails in distr.fails_per_logical:
            lep = log_fails / distr.shots
            assert lep > 0

            # rse calculation
            rse_log = np.sqrt(lep * (1-lep) / distr.shots) / lep
            assert rse_log <= rse_target


def test_file_paths_lists_output_files(all_fail_decoder_managers, tmp_path):
    expr_name = "test_experiment"
    engine = RunAllAnalysisEngine(
        expr_name,
        max_shots=100,
        decoder_managers=all_fail_decoder_managers,
        num_parallel_processes=1,
        data_directory=tmp_path
    )
    engine.run()
    extra_experiment = "test_experiment_2.csv"
    engine.save_results([{}], tmp_path / extra_experiment)
    assert engine.file_paths == [tmp_path / f"{expr_name}.csv",
                                 tmp_path / extra_experiment]


def test_correct_number_of_rows_saved(tmp_path, all_fail_decoder_managers):
    expr_name = "test_experiment"

    engine = RunAllAnalysisEngine(
        expr_name,
        max_shots=10,
        decoder_managers=all_fail_decoder_managers,
        num_parallel_processes=1,
        data_directory=tmp_path
    )
    engine.run()

    if len(all_fail_decoder_managers) > 0:
        saved_data = pd.read_csv(engine.file_paths[0])
        assert saved_data.shape[0] == len(all_fail_decoder_managers)


@pytest.mark.parametrize("failed_shot", [0, 3, 4])
def test_data_saved_until_failure(mocker, tmp_path, failed_shot):
    expr_name = "test_experiment"
    decoder_managers = mock_decoder_managers(mocker, 5, mock_all_fails_run_batch_shots)

    decoder_managers[failed_shot].run_batch_shots.side_effect = ValueError(
        "Decoding interrupted")

    engine = RunAllAnalysisEngine(
        expr_name,
        max_shots=100,
        decoder_managers=decoder_managers,
        num_parallel_processes=1,
        data_directory=tmp_path
    )
    with pytest.raises(ValueError):
        engine.run()

    saved_data = pd.read_csv(engine.file_paths[0])
    assert saved_data.shape[0] == failed_shot
