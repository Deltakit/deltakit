"""
Utilities used by tool scripts
"""

from pathlib import Path

import tomlkit


def extract_version() -> str:
    """
    Extract the version from the pyproject.toml file in the given project path.

    Returns:
        The version string from the project's pyproject.toml

    Note:
        This function will raise:
          - FileNotFoundError: If pyproject.toml does not exist
          - KeyError: If version is not found in pyproject.toml
    """
    proj_home = Path(__file__).parents[1]

    pyproject_path = proj_home / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        pyproject_data: dict = tomlkit.load(f)

    return pyproject_data["project"]["version"]
