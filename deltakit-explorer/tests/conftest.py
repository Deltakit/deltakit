# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import os

from deltakit_explorer._utils._utils import DELTAKIT_SERVER_URL_ENV


def pytest_sessionstart(session):  # noqa: ARG001
    os.environ[DELTAKIT_SERVER_URL_ENV] = "http://deltakit-explorer:8000"
