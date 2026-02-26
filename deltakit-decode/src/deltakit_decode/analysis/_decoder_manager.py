# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from functools import cached_property, partial
from itertools import islice
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4
from warnings import warn

from typing_extensions import override

from deltakit_decode._base_reporter import BaseReporter
from deltakit_decode.analysis._empirical_decoding_error_distribution import (
    EmpiricalDecodingErrorDistribution,
)
from deltakit_decode.noise_sources._generic_noise_sources import (
    BatchErrorGenerator,
    MonteCarloNoise,
    NoiseModel,
    SequentialNoise,
)

ErrorT = TypeVar('ErrorT')
CodeDataT = TypeVar('CodeDataT')
CorrectionT = TypeVar('CorrectionT')
BatchErrorT = TypeVar('BatchErrorT')
BatchCorrectionT = TypeVar('BatchCorrectionT')

mp_dm: DecoderManager | None = None  # pylint: disable=global-statement


class DecoderManager(ABC):
    """Class for the abstract DecoderManger, objects that manage the running of a
    decoder for multiple shots, whilst reporting on the failure rate and running other
    reporters.

    Generally this class contains all the information to generate errors,
    convert to syndromes, decode the result and work out if the decoding was a success.
    Along with support for reporters to log different types of analysis throughout.

    Parameters
    ----------
    reporters : Optional[List[BaseReporter]], optional
        Optional list of reporters to run alongside the decoder, by default None.
    metadata : Optional[Dict[str, str]], optional
        Optional metadata to associate with this decoder manager, by default None.
    reporters : Optional[List[BaseReporter]], optional
        Optional list of reporters to run alongside the decoder, by default None.
    metadata : Optional[Dict[str, Any]], optional
        Optional metadata to associate with this decoder manager, by default None.
    batch_size: int
        The size of batches to run decoding experiments on, by default 10,000.
    number_of_observables: int
        The number of observables to track decoding statistics on, by default 1
        which defines the experiment shot/fail counters.
    """

    def __init__(self,
                 number_of_observables: int,
                 reporters: list[BaseReporter] | None = None,
                 metadata: dict[str, Any] | None = None,
                 batch_size: int = 10000):
        self.reporters = reporters if reporters is not None else []
        self.metadata = metadata if metadata is not None else {}
        self.batch_size = batch_size
        self._number_of_observables = number_of_observables
        self._empirical_decoding_error_distribution = EmpiricalDecodingErrorDistribution(
            self._number_of_observables)
        self._generate_mp_token()

    @abstractmethod
    def run_single_shot(self) -> bool:
        """Run a single shot of decoding. Return True if the decoding shot is deemed
        to have failed.
        """

    @abstractmethod
    def run_batch_shots(self, batch_limit: int | None) -> tuple[int, int]:
        """Run multiple shots of decoding. Return the number of shots ran and the number
        of those that were deemed failures.
        """

    def _setup_process(self):
        """Update state/data of process to this decoder manager.
        """
        global mp_dm  # pylint: disable=global-statement
        mp_dm = self

    def _generate_mp_token(self):
        """Generate token used for batch parallelisation. This token is used to verify
        that worker threads have a consistent decoder manager object with the main
        process decoder manager.
        """
        self._mp_token = uuid4()

    def configure_pool(self, pool, total_processes: int):
        """Configure pool workers to state of this decoder manager.

        Parameters
        ----------
        total_processes : int
            The total number of processes within the pool.
        pool : pathos.multiprocessing.ProcessPool
            Pathos pool to use to run shots in parallel. Assumed given pool is already
            configured with decoder manager data.

        """
        self._generate_mp_token()
        pool.map(lambda _: self._setup_process(), range(total_processes))

    @staticmethod
    def clear_pool_manager(pool, total_processes: int):
        """Configure pool workers to state of this decoder manager.

        Parameters
        ----------
        total_processes : int
            The total number of processes within the pool.
        pool : pathos.multiprocessing.ProcessPool
            Pathos pool to use to run shots in parallel. Assumed given pool is already
            configured with decoder manager data.

        """
        pool.map(lambda _: _clear_process_global_memory(), range(total_processes))

    @abstractmethod
    def run_batch_shots_parallel(self,
                                 batch_limit: int | None,
                                 processes: int,
                                 pool,
                                 min_tasks_per_process: int = 502
                                 ) -> tuple[int, int]:
        """Run batch of shots in parallel using `processes` number of runners from
        `pool`. Cap the number of processes created such that a minimum of
        `min_tasks_per_process` should be running per process. By default the ABC
        implementation is a naive serial execution of run_batch_shots which should
        be overridden for better parallel resource utilisation. This method will block
        until it is finished.

        Each process in the pool should be initialised with this decoder manager stored
        in the process global variables by calling `configure_pool`. The Manager
        authenticates that a worker processes has a consistent decoder manager with the
        main process by comparing the managers `_mp_token`. The pool should have its
        state cleared once a sequence of calls to this method exclusive to this decoder
        manager is finished by calling `clear_pool_manager`.

        Parameters
        ----------
        batch_limit : Optional[int]
            How many samples to take. If None, will attempt to exhaust the noise source
        processes : int
            How many runners to distribute shot execution across.
        pool : pathos.multiprocessing.ProcessPool
            Pathos pool to use to run shots in parallel. Assumed given pool is already
            configured with decoder manager data.
        min_shots_per_process: int, optional
            Minimum of shots that should be running per process, will reduce `processes`
            if umber of shots taken and number of those shots that failed.

        Example
        --------
        .. code-block:: python

            num_parallel_processes = 4
            decoder_manager.configure_pool(pool, num_parallel_processes)
            for i in range(10):
                decoder_manager.run_batch_shots_parallel(
                    batch_limit=self.batch_size,
                    processes=num_parallel_processes,
                    pool=pool
                )
            decoder_manager.clear_pool_manager(pool, num_parallel_processes)
        """
        return self.run_batch_shots(batch_limit=batch_limit)

    def get_reporter_results(self) -> dict[str, Any]:
        """Get aggregated data from the manager's internal state and all available
        reporters as a dict from string of data identifier to data.

        Returns
        -------
        Dict[str, Any]
            Dictionary of current reporting results.
        """
        analysis_results: dict[str, Any] = {"shots": self.shots, "fails": self.fails}
        analysis_results.update(self.metadata)

        for reporter in self.reporters:
            for key, val in reporter.get_reported_results().items():
                if key not in analysis_results:
                    analysis_results[key] = val
                else:
                    warn("Warning: at least 2 reporters tried to report information "
                         "using the same key. Only the first reporter information will "
                         "be included in the returned results.",
                         stacklevel=2)

        return analysis_results

    def reset(self):
        """Reset all reporters and their aggregations. Reset empirical
        decoding error distribution.
        """
        for reporter in self.reporters:
            reporter.reset_reporter()
        self._empirical_decoding_error_distribution.reset()

    @property
    def shots(self) -> int:
        """ Number of decoding shots run since last reset.
        """
        return self._empirical_decoding_error_distribution.shots

    @property
    def fails(self) -> int:
        """ Number of failed decodes since last reset.
        """
        return self._empirical_decoding_error_distribution.fails

    @property
    def empirical_decoding_error_distribution(
        self,
    ) -> EmpiricalDecodingErrorDistribution:
        """Information of all decoding events since last reset.
        """
        return self._empirical_decoding_error_distribution


class InvalidGlobalManagerStateError(ValueError):
    """Raise when decoder manager is not set in global memory or
    has invalid state token.
    """


def _clear_process_global_memory():
    """Clear process decoder manager data in globals memory.
    """
    global mp_dm  # pylint: disable=global-statement
    mp_dm = None


def _run_process(state_token: UUID, batch_limit: int | None, jobs: int, job_no: int
                 ) -> tuple[EmpiricalDecodingErrorDistribution,
                            list[BaseReporter]]:
    """Run process with persistent decoder manager object stored in process
    globals memory.
    """
    if isinstance(mp_dm, NoiseModelDecoderManager):
        result = mp_dm._thread_worker(state_token,  # pylint: disable=protected-access
                                      batch_limit,
                                      jobs,
                                      job_no)
    else:
        msg = "Process state not configured with decoder manager."
        raise InvalidGlobalManagerStateError(msg)
    return result


class NoiseModelDecoderManager(DecoderManager,
                               Generic[ErrorT, CodeDataT, CorrectionT,
                                       BatchErrorT, BatchCorrectionT]):
    """Decoder managers that use a specific interface for running the decoder enabled by
    the use of a `NoiseModel` object.

    Generic over `ErrorT` for the type of error decoders run over. `CodeDataT` for the
    type that is needed as input for the `NoiseModel` and `CorrectionT` for the type of
    correction that decoders return.

    Parameters
    ----------
    noise_model : NoiseModel[CodeDataT, ErrorT]
        NoiseModel that this decoder manager will use to generate errors and syndromes.
    reporters : Optional[List[BaseReporter]], optional
        Optional list of reporters to run alongside the decoder, by default None.
    metadata : Optional[Dict[str, str]], optional
        Optional metadata to associate with this decoder manager, by default None.
    seed : Optional[int], optional
        Optional seed for use with given noise sources, by default None which results in
        random seed generation.
    """

    def __init__(self,
                 noise_model: NoiseModel[CodeDataT, ErrorT],
                 number_of_observables: int,
                 reporters: list[BaseReporter] | None = None,
                 metadata: dict[str, Any] | None = None,
                 seed: int | None = None,
                 batch_size: int = int(1e4)):
        super().__init__(number_of_observables,
                         reporters,
                         metadata,
                         batch_size=batch_size)
        self._noise_model = noise_model
        self._shared_generator: Iterator[ErrorT] | None = None
        self._seed = seed
        self._start_seed = seed

    def run_single_shot(self) -> bool:
        with ExitStack() as stack:
            for reporter in self.reporters:
                stack.enter_context(reporter)
            error = self.generate_single_error()
            correction = self._decode_from_error(error)
            return self._analyse_correction(error, correction)


    def _exec_shots_atomic(self,
                           error_generator: Iterable[ErrorT],
                           batch_limit: int | None):
        """Executes shot one by one."""
        if batch_limit is not None:
            error_generator = islice(error_generator, int(batch_limit))
        elif isinstance(self._noise_model, MonteCarloNoise):
            msg = (
                f"Cannot use MonteCarlo noise source {self._noise_model} without a "
                "specified batch_limit."
            )
            raise ValueError(msg)

        for error in error_generator:
            with ExitStack() as stack:
                for reporter in self.reporters:
                    stack.enter_context(reporter)
                correction = self._decode_from_error(error)
                self._analyse_correction(error, correction)

    def _exec_shots_batch(self,
                          batch_error_generator: BatchErrorGenerator,
                          batch_limit: int):
        """Executes batches of shots."""
        if self.reporters:
            msg = "Can not run batches with reporters!"
            raise ValueError(msg)

        batch_num = int(batch_limit // self.batch_size)
        remaining_shots = batch_limit - (batch_num * self.batch_size)
        batch_sizes = [self.batch_size] * batch_num

        if remaining_shots > 0:
            batch_sizes.append(remaining_shots)

        for batch_size in batch_sizes:
            error = batch_error_generator(batch_size)
            self._decode_batch_from_error(error)

    def reset(self):
        self._seed = self._start_seed
        self.__dict__.pop('batch_error_generator', None)
        self.__dict__.pop('error_generator', None)
        super().reset()

    @cached_property
    def batch_error_generator(self) -> BatchErrorGenerator:
        """Gets and caches batch error generator given the code data."""
        return self._noise_model.build_batch_error_generator(self._get_code_data(),
                                                             seed=self._seed)

    @cached_property
    def error_generator(self) -> Iterable[ErrorT]:
        """Gets and caches error generator given the code data."""
        return self._noise_model.error_generator(
            self._get_code_data(), seed=self._seed)

    def run_batch_shots(self, batch_limit: int | None) -> tuple[int, int]:
        if batch_limit is not None and batch_limit > 1 and not self.reporters:
            self._exec_shots_batch(self.batch_error_generator, batch_limit)
        else:
            self._exec_shots_atomic(self.error_generator, batch_limit)
        return self.shots, self.fails

    @override
    def run_batch_shots_parallel(self,
                                 batch_limit: int | None,
                                 processes: int,
                                 pool,
                                 min_tasks_per_process: int = 50
                                 ) -> tuple[int, int]:
        if batch_limit is None:
            if isinstance(self._noise_model, MonteCarloNoise):
                msg = (
                    f"Cannot use MonteCarlo noise source {self._noise_model} without a "
                    "specified batch_limit."
                )
                raise ValueError(msg)
            task_num = self._noise_model.sequence_size(self._get_code_data())
        else:
            task_num = int(batch_limit)

        processes = min((task_num // min_tasks_per_process) + 1, processes)
        inner_batch_limit = int(batch_limit) // processes if batch_limit is not None \
            else batch_limit
        # Attempt to run batches in parallel. If the pool has caused a process to lose
        # the decoder manager in its globals memory, then reinitialise processes with
        # the decoder manager object and run batches in the same process call.
        try:
            partial_worker = partial(_run_process, self._mp_token,
                                     inner_batch_limit, processes)
            results = pool.map(partial_worker, range(processes))
        except InvalidGlobalManagerStateError:
            partial_worker = partial(self._setup_and_run_process,
                                     inner_batch_limit, processes)
            results = pool.map(partial_worker, range(processes))

        for (distr, reporters) in results:
            self._empirical_decoding_error_distribution += distr
            for i, reporter in enumerate(reporters):
                self.reporters[i] += reporter
        if self._seed is not None:
            self._seed += hash(NoiseModelDecoderManager)
        return self.shots, self.fails

    def generate_single_error(self) -> ErrorT:
        """Generate an error for a single shot of decoding.
        """
        if isinstance(self._noise_model, SequentialNoise):
            warn(f"Taking single error from {self._noise_model} may result in "
                 "unexpected behaviour, consider using `run_batch_shots`",
                 stacklevel=2)
        if self._shared_generator is None:
            self._shared_generator = self._noise_model.error_generator(
                self._get_code_data(), seed=self._seed)
        return next(self._shared_generator)

    def _setup_and_run_process(self, batch_limit: int | None, jobs: int, job_no: int
                               ) -> tuple[EmpiricalDecodingErrorDistribution,
                                          list[BaseReporter]]:
        """Setup process globals memory with manager and run thread worker.
        """
        global mp_dm  # pylint: disable=global-statement
        mp_dm = self  # pylint: disable=global-statement
        if job_no < jobs:
            result = self._thread_worker(self._mp_token,
                                         batch_limit,
                                         jobs,
                                         job_no)  # pylint: disable=protected-access
        else:
            # if job_no surpasses requested number of jobs return empty distribution
            self.empirical_decoding_error_distribution.reset()
            result = self._empirical_decoding_error_distribution, self.reporters
        return result

    def _thread_worker(self,
                       state_token,
                       batch_limit: int | None,
                       jobs: int, job_no: int
                       ) -> tuple[EmpiricalDecodingErrorDistribution,
                                  list[BaseReporter]]:
        if state_token != self._mp_token:
            msg = "Worker decoder manager global has invalid token."
            raise InvalidGlobalManagerStateError(msg)
        self.reset()

        seed = self._seed if self._seed is None else self._seed + job_no
        if batch_limit is not None and not self.reporters:
            batch_error_generator: BatchErrorGenerator
            batch_error_generator, _ = (
                self._noise_model.build_split_batch_error_generators(
                    self._get_code_data(), num_splits=jobs, seed=self._seed)[job_no])
            self._exec_shots_batch(batch_error_generator, batch_limit)
        else:
            error_generator: Iterable[ErrorT]
            error_generator, _ = self._noise_model.split_error_generator(
                self._get_code_data(), num_splits=jobs, seed=seed)[job_no]
            self._exec_shots_atomic(error_generator, batch_limit)

        return self._empirical_decoding_error_distribution, self.reporters

    @abstractmethod
    def _get_code_data(self) -> CodeDataT:
        """Private method to get the code data needed by the `NoiseModel` to generate
        errors for a particular code.
        """

    @abstractmethod
    def _analyse_correction(self, error: ErrorT, correction: CorrectionT) -> bool:
        """Private method to take the result of a decoder and the error it decoded
        and analyse the correctness of the decoder output.

        Parameters
        ----------
        error : ErrorT
            Error that was decoded.
        correction : CorrectionT
            The correction returned by the decoder.

        Returns
        -------
        bool
            Return True if the decoder failed, False otherwise.
        """

    @abstractmethod
    def _decode_from_error(self, error: ErrorT) -> CorrectionT:
        """Private method to take an error, convert it to the form that can be
        decoded, and then return the correction output by the decoder.
        """

    @abstractmethod
    def _decode_batch_from_error(self, errors: BatchErrorT) -> BatchCorrectionT:
        """Private method to take an error, convert it to the form that can be
        decoded, and then return the correction output by the decoder.
        """
