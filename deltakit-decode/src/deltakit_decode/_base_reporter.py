# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import logging
from abc import abstractmethod
from contextlib import AbstractContextManager
from math import sqrt
from time import time_ns
from typing import Any

from typing_extensions import Self

from deltakit_decode.utils import make_logger


class BaseReporter(AbstractContextManager):
    """Base class for reporters. Reporters act as a context manager for
    the execution of decoders. They store and aggregate various metrics
    about the decoder that is running.

    Parameters
    ----------
    lvl : int, optional
        Log level to be used, by default logging.ERROR.
    """

    def __init__(self, lvl: int = logging.ERROR):
        self.log = make_logger(lvl, self.__class__.__name__)

    @abstractmethod
    def reset_reporter(self):
        """Reset reporter back to initial state, clearing all metrics and
        aggregates.
        """

    @abstractmethod
    def get_reported_results(self) -> dict[str, Any]:
        """Return the usage log values this reporter has recorded.
        This is given as a dictionary where the key is some string describing the data.
        """

    def __iadd__(self, other: BaseReporter) -> Self:
        """Add outcomes from another reporter of the same kind.
        This is necessary for collecting results from multiple reporters
        when using parallel processes.
        """
        msg = "Reporter concatenation not defined!"
        raise ValueError(msg)


class TimingReporter(BaseReporter):
    """Simple reporter that records the total amount
    of time spent decoding, in nanoseconds.
    """

    def __init__(self, lvl: int = logging.ERROR):
        super().__init__(lvl)
        # Metrics
        self._start_time: int = 0
        self._end_time: int = 0

        # Aggregates
        self._sum_wall_ns: int = 0
        self._shots: int = 0
        # Welford's online variance algorithm
        # \bar x_n = mean of first n values
        # \sum_{i=1}^n (x_i - \bar x_n)^2
        self._sum_of_square_deviations: float = 0

    def __str__(self) -> str:
        return "TimingReporter"

    def __enter__(self):
        self._start_time = time_ns()

    def __exit__(self, exc_type, exc_value, traceback):
        self._end_time = time_ns()
        last_shot_wall_ns = self._end_time - self._start_time
        if self._shots > 0:
            # here this is the old shot count, so this is true except on the first shot
            xn_minus_oldmean = last_shot_wall_ns - self._sum_wall_ns / self._shots
        else:
            xn_minus_oldmean = 0
        self._sum_wall_ns += last_shot_wall_ns
        self._shots += 1
        xn_minus_newmean = last_shot_wall_ns - self._sum_wall_ns / self._shots
        self._sum_of_square_deviations += xn_minus_oldmean * xn_minus_newmean

    def reset_reporter(self):
        self._sum_wall_ns = 0
        self._shots = 0
        self._sum_of_square_deviations = 0
        self._start_time = self._end_time = 0

    @property
    def avg_wall_ns(self):
        """Average nanoseconds of wall clock time taken to decode per shot
        """
        if self._shots > 0:
            return self._sum_wall_ns / self._shots
        return 0

    @property
    def stderr_wall_ns(self):
        """Standard deviation in nanoseconds of wall clock time taken to decode per shot
        """
        if self._shots > 1:
            return sqrt(self._sum_of_square_deviations / (self._shots - 1))
        return 0

    def get_reported_results(self) -> dict[str, Any]:
        return {
            "avg_wall_ns": self.avg_wall_ns,
            "stderr_wall_ns": self.stderr_wall_ns
        }

    def __iadd__(self, other: BaseReporter) -> Self:
        if isinstance(other, TimingReporter):
            # use parallel Welford's algorithm formula
            total_shots = self._shots + other._shots
            delta = other.avg_wall_ns - self.avg_wall_ns
            self._sum_of_square_deviations = \
                self._sum_of_square_deviations + \
                other._sum_of_square_deviations + \
                delta**2 * self._shots * other._shots / total_shots
            self._sum_wall_ns += other._sum_wall_ns
            self._shots = total_shots
            return self
        return NotImplemented
