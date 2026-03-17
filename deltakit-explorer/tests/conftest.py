# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import os

import numpy as np
import pytest

from deltakit_explorer._utils._utils import DELTAKIT_SERVER_URL_ENV
from deltakit_explorer.analysis import LambdaData
from deltakit_explorer.analysis import (
    LogicalErrorProbabilityPerRoundData as LEPPRData,
)


def pytest_sessionstart(session):  # noqa: ARG001
    os.environ[DELTAKIT_SERVER_URL_ENV] = "http://deltakit-explorer:8000"


@pytest.fixture(scope="session")
def random_generator():
    return np.random.default_rng()


@pytest.fixture
def lambda_results() -> LambdaData:
    return LambdaData(
        lambda_=3.0,
        lambda_stddev=0.1,
        lambda0=1.5,
        lambda0_stddev=0.05,
    )


@pytest.fixture
def leppr_results() -> LEPPRData:
    return LEPPRData(
        leppr=0.001,
        leppr_stddev=0.0001,
        spam_error=0.01,
        spam_error_stddev=0.001,
    )


@pytest.fixture
def distances() -> np.ndarray:
    return np.array([5, 7, 9])


@pytest.fixture
def num_rounds() -> np.ndarray:
    return np.array([2, 4, 6])
