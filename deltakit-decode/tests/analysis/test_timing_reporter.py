# (c) Copyright Riverlane 2020-2025.
from contextlib import ExitStack
from itertools import cycle
from unittest.mock import Mock

import pytest

from deltakit_decode._base_reporter import TimingReporter


@pytest.fixture
def timing_reporter():
    return TimingReporter()


@pytest.fixture(autouse=True)
def mock_time_ns(monkeypatch):
    mock = Mock()
    mock.side_effect = cycle([0, 1])
    monkeypatch.setattr("deltakit_decode._base_reporter.time_ns", mock)


def test_single_shot(timing_reporter: TimingReporter):
    with ExitStack() as stack:
        stack.enter_context(timing_reporter)
    reported_results = timing_reporter.get_reported_results()
    assert reported_results == {"avg_wall_ns": 1, "stderr_wall_ns": 0}


def test_multiple_shots(timing_reporter: TimingReporter):
    for _ in range(5):
        with ExitStack() as stack:
            stack.enter_context(timing_reporter)
    reported_results = timing_reporter.get_reported_results()
    assert reported_results == {"avg_wall_ns": 1.0, "stderr_wall_ns": 0}


def test_adding_reporters(timing_reporter: TimingReporter):
    timing_reporter2 = TimingReporter()
    for _ in range(5):
        with ExitStack() as stack:
            stack.enter_context(timing_reporter)
    for _ in range(5):
        with ExitStack() as stack:
            stack.enter_context(timing_reporter2)
    timing_reporter += timing_reporter2
    reported_results = timing_reporter.get_reported_results()
    assert timing_reporter._sum_wall_ns == 10.0
    assert timing_reporter._shots == 10
    assert reported_results == {"avg_wall_ns": 1.0, "stderr_wall_ns": 0}


def test_reset_reporter(timing_reporter: TimingReporter):
    with ExitStack() as stack:
        stack.enter_context(timing_reporter)
    timing_reporter.reset_reporter()
    reset_reported_results = timing_reporter.get_reported_results()
    assert reset_reported_results == {"avg_wall_ns": 0, "stderr_wall_ns": 0}


def test_str(timing_reporter: TimingReporter):
    with ExitStack() as stack:
        stack.enter_context(timing_reporter)
    assert str(timing_reporter) == "TimingReporter"


def test_add(timing_reporter: TimingReporter):
    with ExitStack() as stack:
        stack.enter_context(timing_reporter)
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for \+="):
        timing_reporter += 1  # type: ignore[arg-type]
