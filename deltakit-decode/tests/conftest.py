# (c) Copyright Riverlane 2020-2025.
from pathlib import Path

import numpy as np
import pytest
from _pytest.python import Function, Metafunc, Parser


def pytest_addoption(parser: Parser):
    """Hook function called by pytest to add command line options for this particular repo.

    Parameters
    ----------
    parser : Parser
        A pytest parser instance to add command line options to.
    """
    parser.addoption(
        "--random-repeats",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        action="store",
    )


def pytest_generate_tests(metafunc: Metafunc):
    """Hook function called by pytest to generate the tests. This is run at collection time before
    any tests have run. This function creates a small wrapper around `pytest-repeat` to generate a
    number of tests based on the command line options.

    Parameters
    ----------
    metafunc : Metafunc
        This method is called for each test function that pytest collects and is passed a `Metafunc`
        object which exposes a number of extra testing attributes.
    """
    repeat_options = {"low": 1, "medium": 10, "high": 50}
    for marker in reversed(list(metafunc.definition.iter_markers(name="random"))):
        repeat_options.update(marker.kwargs)
    if metafunc.definition.get_closest_marker("random") is not None:
        number_of_repeats = repeat_options[
            metafunc.config.getoption("--random-repeats")
        ]
        metafunc.definition.add_marker(pytest.mark.repeat(number_of_repeats))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Function):
    """Hook function called when running the setup of a test. This checks if the test is marked
    as `skip_if_low` and skips the test if the coverage option is `low` otherwise runs the test.

    Parameters
    ----------
    item : Function
        A pytest function object responsible for running setup, test and teardown.
    """
    coverage_option = item.config.getoption("--random-repeats")
    if (marker := item.get_closest_marker("skip_if_low")) and coverage_option == "low":
        pytest.skip(marker.kwargs.get("reason"))  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def git_root():
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def random_generator():
    return np.random.default_rng()


@pytest.fixture(scope="session")
def reference_data_dir(git_root: Path):
    return git_root / "tests" / "reference_data"
