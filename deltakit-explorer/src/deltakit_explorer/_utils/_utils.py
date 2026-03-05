# (c) Copyright Riverlane 2020-2025.
"""
Contains a few utility functions to be used in deltakit api
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

APP_NAME = "deltakit-explorer"
DELTAKIT_SERVER_URL_ENV = "DELTAKIT_SERVER"
DELTAKIT_SERVER_DEFAULT_URL_ENV = "https://deltakit.riverlane.com/proxy"
HTTP_PACKET_LIMIT = 20_000_000


def get_log_directory() -> Path:
    """
    Get or generate the log directory path for the user.

    Returns:
        Path: The path to the log directory based on the order of availability.
    """

    # Check for 'LOG_DIRECTORY' environment variable
    env_folder = os.getenv("LOG_DIRECTORY")
    if env_folder:
        env_data_folder = Path(env_folder)
        if env_data_folder.exists() and env_data_folder.is_dir():
            return Path(env_folder)

    return Path.cwd()


def get_config_directory() -> Path:
    """Try to obtain OS-specific configuration folder for an app.
    Fallback scenario is to use current working directory."""
    try:
        config_dir_path, override_path = None, None
        if sys.platform == "win32":
            config_dir_path = Path.home() / "AppData" / "Local" / APP_NAME
            override_path = os.getenv("APPDATA")
        elif sys.platform == "darwin":
            config_dir_path = Path.home() / "Library" / "Application Support" / APP_NAME
        else:  # Linux and other Unix-like
            config_dir_path = Path.home() / ".config" / APP_NAME
            override_path = os.getenv("XDG_CONFIG_HOME")

        if override_path is not None:
            config_dir_path = Path(override_path) / APP_NAME
        if not config_dir_path.exists():
            config_dir_path.mkdir(parents=True)
        return config_dir_path
    except PermissionError:
        return Path.cwd()


def get_config_file_path() -> Path:
    """Path to a config file. Ensure existence of the folder."""
    return get_config_directory() / ".env"


def read_persisted_variables(file: Path) -> dict[str, str]:
    """
    Given a filename, returns content as a dictionary.
    Separator of key and value is `=`.
    Separator of items is a new line.

    Args:
        file (Path): path to a file with variables.

    Returns:
        Dict[str, str]:
            dictionary of persisted variables.
    """
    result = {}
    if file.exists():
        with Path.open(file, encoding="utf-8") as csv_stream:
            for row in csv.reader(csv_stream, delimiter="="):
                # empty line is ok
                if len(row) == 0:
                    continue
                # parsing of env file failed
                if len(row) != 2:
                    return {}
                result[row[0].strip()] = row[1].strip()
    return result


def set_variables(variables: dict[str, str], override: bool = False):
    """
    Set environment variables. If variable is defined,
    do not rewrite.

    Args:
        variables (Dict[str, str]): new values for environment variables.
        override (bool): override the variable.
    """
    for key, value in variables.items():
        if key not in os.environ or override:
            os.environ[key] = value


def load_environment_variables_from_drive():
    """
    If any environment variables are persisted, load them
    without overriding passed ones.
    """
    file = get_config_file_path()
    variables = read_persisted_variables(file)
    set_variables(variables, False)


def override_persisted_variables(variables: dict[str, str], file: Path):
    """
    Sets variables to the file. Overrides existing file.

    Args:
        variables (Dict[str, str]): new values to save in the file.
    """
    with Path.open(file, "w", encoding="utf-8") as csv_stream:
        writer = csv.writer(csv_stream, delimiter="=")
        for key, value in variables.items():
            writer.writerow([key, value])


def merge_variables(variables: dict[str, str], file: Path):
    """
    Adds or overrides the values in the file. Keeps old values.

    Args:
        variables (Dict[str, str]): new values to save in the file.
    """
    old_variables = read_persisted_variables(file)
    old_variables.update(variables)
    override_persisted_variables(old_variables, file)
