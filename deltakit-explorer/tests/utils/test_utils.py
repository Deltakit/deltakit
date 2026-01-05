# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from deltakit_explorer._utils import _utils
from deltakit_explorer._utils._utils import get_log_directory


@pytest.fixture
def reset_env():
    """Fixture to clear the LOG_DIRECTORY environment variable after each test."""

    old_env = os.environ.get("LOG_DIRECTORY")
    yield
    if old_env:
        os.environ["LOG_DIRECTORY"] = old_env
    else:
        os.environ.pop("LOG_DIRECTORY", None)

def test_log_directory_env_variable(tmp_path):
    """Set LOG_DIRECTORY environment variable to a temporary path"""

    temp_dir = tmp_path / "log_dir_env"
    temp_dir.mkdir()
    os.environ["LOG_DIRECTORY"] = str(temp_dir)

    result = get_log_directory()
    assert result == temp_dir
    assert result.is_dir()


def test_log_directory_home_directory(tmp_path):
    """Ensure LOG_DIRECTORY env variable is not set"""

    os.environ.pop("LOG_DIRECTORY", None)

    with mock.patch.object(Path, "cwd", return_value=tmp_path):
        cwd = Path.cwd()

        # Verify function uses the 'data' directory in the user's home
        result = get_log_directory()
        assert result == cwd
        assert result.is_dir()

def test_log_directory_env_variable_not_a_directory(tmp_path):
    """Set LOG_DIRECTORY to a file path instead of a directory"""

    temp_file = tmp_path / "log_file.txt"
    temp_file.write_text("This is a file, not a directory")
    os.environ["LOG_DIRECTORY"] = str(temp_file)

    with mock.patch.object(Path, "cwd", return_value=tmp_path):
        Path.cwd()

        result = get_log_directory()
        assert result == Path.cwd()
        assert result.is_dir()

def test_log_directory_env_variable_invalid_path():
    """Set LOG_DIRECTORY to an invalid path string"""

    os.environ["LOG_DIRECTORY"] = "/invalid_path/!@#"

    tmp = Path(tempfile.mkdtemp())
    ret = os.getenv("TEMP", tmp)
    try:
        with mock.patch.object(Path, "cwd", return_value=Path(ret)):
            cwd = Path.cwd()

            result = get_log_directory()
            assert result == cwd
            assert result.is_dir()
    except Exception:
        tmp.rmdir()
        raise

def test_env_and_home_directory_unavailable():
    """Ensure LOG_DIRECTORY env variable is not set"""

    os.environ.pop("LOG_DIRECTORY", None)

    with mock.patch.object(Path, "home", side_effect=PermissionError), \
         mock.patch.object(Path, "cwd", side_effect=OSError), pytest.raises(OSError):
        get_log_directory()


def test_read_persisted_variables(tmp_path):
    file = tmp_path / "vars.env"
    file.write_text("A=1\n\n\nB=2\n\n")
    result = _utils.read_persisted_variables(file)
    assert result["A"] == "1"
    assert result["B"] == "2"

def test_read_persisted_variables_invalid(tmp_path):
    file = tmp_path / "vars.env"
    file.write_text("A=1\nB\n")
    result = _utils.read_persisted_variables(file)
    assert result == {}

def test_merge_variables(tmp_path):
    file = tmp_path / "vars.env"
    file.write_text("A=1\n")
    _utils.merge_variables({"B": "2"}, file)
    result = _utils.read_persisted_variables(file)
    assert result["B"] == "2"
