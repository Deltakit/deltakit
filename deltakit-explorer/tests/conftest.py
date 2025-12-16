# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import os

import numpy as np
import pytest

from deltakit_explorer._utils._utils import DELTAKIT_SERVER_URL_ENV


def pytest_sessionstart(session):  # noqa: ARG001
    os.environ[DELTAKIT_SERVER_URL_ENV] = "http://deltakit-explorer:8000"


@pytest.fixture(scope="session")
def random_generator():
    return np.random.default_rng()
