# (c) Copyright Riverlane 2020-2025.

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def git_root():
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def reference_data_dir(git_root: Path):
    return git_root / "tests" / "reference_data"
