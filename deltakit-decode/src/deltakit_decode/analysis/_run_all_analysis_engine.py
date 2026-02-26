# (c) Copyright Riverlane 2020-2025.
import datetime
import logging
from collections.abc import Callable, Iterable
from multiprocessing.synchronize import Lock as LockBase
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pathos.pools import ProcessPool
from tqdm import tqdm

from deltakit_decode.analysis._decoder_manager import DecoderManager
from deltakit_decode.analysis._empirical_decoding_error_distribution import (
    EmpiricalDecodingErrorDistribution,
)
from deltakit_decode.utils import make_logger


class RunAllAnalysisEngine:
    """Class to run experiments on different codes and decoders, it will run
    all of the given sequence of decoder managers.

    Parameters
    ----------
    experiment_name : str
        Name of this experiment.
    loop_condition : Callable[EmpiricalDecodingErrorDistribution, bool]
        For a given code and noise model, this loop condition controls how
        many evaluations of a particular code and noise model are run.
        It is given as a function that takes in the current empirical decoding
        error distribution and returns True if another shot should be performed.
    data_directory : Optional[Path], optional
        Path to directory in which to store experiment results data. If not
        given, no data is output to CSV. By default None.
    num_parallel_processes : int, optional
        Number of parallel processes to use. If <=1, process will be run sequentially.
        By default 16.
    lvl : int, optional
        Logging level to use, by default logging.NOTSET.
    batch_size : int, optional
        The number of decoding shots to run in a batch. Should beset as
        high as possible.
    max_shots : int, optional
        The maximum number of shots to run.
    """

    def __init__(
        self,
        experiment_name: str,
        decoder_managers: Iterable[DecoderManager],
        loop_condition: Callable[[EmpiricalDecodingErrorDistribution], bool] | None = None,
        data_directory: Path | None = None,
        num_parallel_processes: int = 16,
        lvl: int = logging.NOTSET,
        batch_size: int = 10000000,
        max_shots: int = 10000000,
    ):
        self.loop_condition = loop_condition
        self.num_parallel_processes = num_parallel_processes
        self.parallel = num_parallel_processes > 1
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.max_shots = max_shots
        if data_directory is not None and not data_directory.is_dir():
            raise NotADirectoryError(data_directory)

        self.data_directory = data_directory
        self.decoder_managers = decoder_managers
        self.log = make_logger(lvl, experiment_name)
        self.file_paths: list[Path] = []
        self._current_experiment_file_path: Path | None = None
        self._running_data = pd.DataFrame()

    @property
    def all_reported_fields(self) -> list[str]:
        """Returns the list of all fields reported by the decoder_managers. These are
        going to be headers of the exported CSV."""
        # using dict instead of set to keep ordering consistent
        reported_fields = {
            field_name: None
            for decoder_manager in self.decoder_managers
            for field_name in decoder_manager.get_reporter_results()
        }
        return list(reported_fields.keys())

    def run(self):
        """For all codes on all noise sources given to this instance,
        evaluate the given decoder and return the number of logical
        failures, non-zero syndromes and stabilisers after several shots
        in accordance with the given loop condition.

        If 'data_directory' exists, results are stored in a csv file.
        Else returns a pandas DataFrame.
        """
        # Experiment setup
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.info("Experiment started at %(time)", extra={"time": now})
        if self.data_directory:
            self._current_experiment_file_path = self.construct_file_path()
            self.file_paths.append(self._current_experiment_file_path)
            pd.DataFrame(columns=self.all_reported_fields).to_csv(
                self._current_experiment_file_path, index=False
            )
        else:
            self._current_experiment_file_path = None

        result_store = self._run_parallel() if self.parallel else self._run_serial()

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.info("Experiment finished at %(time)", extra={"time": now})

        return pd.DataFrame(result_store)

    def _run_parallel(self) -> list[dict[str, Any]]:
        """Helper function to run the decoder managers in parallel using a
        pathos process pool. Returns the list of shot loop results.
        """
        pool = ProcessPool(nodes=self.num_parallel_processes)
        # Assumes decoder managers execution is blocking.
        desc = "Evaluating codes"
        fmt = "{l_bar}{bar}{r_bar}\n" if self.log.level else "{l_bar}{bar}{r_bar}"
        tqdm_iter = tqdm(self.decoder_managers, desc=desc, bar_format=fmt)
        return [self._shot_loop(decoder_manager, pool=pool)
                for decoder_manager in tqdm_iter]

    def _run_serial(self) -> list[dict[str, Any]]:
        """Helper function to run the decoder managers in serial.
        Returns the list of shot loop results.
        """
        desc = "Evaluating codes"
        fmt = "{l_bar}{bar}{r_bar}\n" if self.log.level else "{l_bar}{bar}{r_bar}"
        tqdm_iter = tqdm(self.decoder_managers, desc=desc, bar_format=fmt)
        return [self._shot_loop(decoder_manager) for decoder_manager in tqdm_iter]

    def construct_file_path(self) -> Path:
        """Return the file path to be used for the results data."""
        if not self.data_directory:
            msg = "Cannot construct file path as no data directory is specified."
            raise ValueError(msg)
        return self.data_directory / f"{self.experiment_name}.csv"

    def save_results(self, results: list[dict], file_path: Path):
        """Save the results to file and log that the file path was used."""
        pd.DataFrame(results).to_csv(file_path, index=False)
        self.file_paths.append(file_path)

    def _append_results_to_current_file(self, results: dict[str, Any]):
        """Appends the row of results to the current file"""
        pd.DataFrame([results]).to_csv(
            self._current_experiment_file_path, mode="a", index=False, header=False
        )

    def _shot_loop(
        self, decoder_manager: DecoderManager,
        file_save_lock: LockBase | None = None,
        pool: ProcessPool = None
    ) -> dict[str, Any]:
        """Private helper function for performing a single loop for a given
        noise model, decoder and code. Returns an aggregation of accuracy
        statistics.
        """
        self.log.info(
            "Starting %(decoder_manager) with metadata: %(metadata).",
            extra={
                "decoder_manager": decoder_manager,
                "metadada": decoder_manager.metadata,
            }
        )
        if self.parallel:
            decoder_manager.configure_pool(pool, self.num_parallel_processes)

        while (remaining_shots := int(
            self.max_shots
            - decoder_manager.empirical_decoding_error_distribution.shots)
        ) > 0 and (self.loop_condition is None or self.loop_condition(
                decoder_manager.empirical_decoding_error_distribution)):
            procs_needed = min(self.num_parallel_processes,
                               remaining_shots//self.batch_size)
            if self.parallel and procs_needed > 1:
                decoder_manager.run_batch_shots_parallel(
                    batch_limit=self.batch_size * procs_needed,
                    processes=procs_needed,
                    pool=pool
                )
            else:
                batch_size = min(self.batch_size, remaining_shots)
                decoder_manager.run_batch_shots(batch_size)

        if self.parallel:
            decoder_manager.clear_pool_manager(pool, self.num_parallel_processes)

        self.log.info(
            "Finished %(decoder_manager) with metadata: %(metadata).",
            extra={
                "decoder_manager": decoder_manager,
                "metadada": decoder_manager.metadata,
            },
        )
        results = decoder_manager.get_reporter_results()
        if self._current_experiment_file_path is not None:
            if file_save_lock is not None:
                acquired = file_save_lock.acquire(timeout=2)
                if not acquired:
                    msg = "Lock not acquired - can not write to the file!"
                    raise TimeoutError(msg)
                self._append_results_to_current_file(results)
                file_save_lock.release()
            else:
                self._append_results_to_current_file(results)

        return results

    @staticmethod
    def loop_until_failures(count: int):
        """Loop until there have been at least a specific number of failures, as defined
        by the decoder manager.

        Parameters
        ----------
        count : int
            Number of failures to run.
        """

        def loop_cond(empirical_decoding_error_distribution):
            return empirical_decoding_error_distribution.fails < count

        return loop_cond

    @staticmethod
    def loop_until_observable_rse_below_threshold(target_rse: float, min_fails: int):
        """Loop until the maximum number of shots or relative standard
        error on the estimate of LEP go below the given amount.
        The RSE stopping criteria is only taken into account if at least
        min_fails are reached.

        Parameters
        ----------
        target_rse : Optional[float], optional
            Target Relative Standard Error (RSE) for early stopping.
        min_fails : int, optional
            Minimum number of fails before starting to look at RSE.
        """

        def loop_cond(empirical_decoding_error_distribution):
            rse = 0
            shots = empirical_decoding_error_distribution.shots
            if shots == 0:
                return True

            fails_per_observable = (
                empirical_decoding_error_distribution.fails_per_logical
            )

            min_fails_any_observable = fails_per_observable[0]
            for log_fails in fails_per_observable:
                lep = log_fails / shots
                min_fails_any_observable = min(min_fails_any_observable, log_fails)

                if lep == 0:
                    continue

                rse_log = np.sqrt(lep * (1 - lep) / shots) / lep
                rse = max(rse, rse_log)
            return rse > target_rse or min_fails_any_observable < min_fails

        return loop_cond
