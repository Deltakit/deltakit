"""
Script to set prerelease version number in all `pyproject.toml`s.
Usage: `python tools/set_pre_version.py <suffix>`
e.g.  `python tools/set_pre_version.py .dev20250820160500`
"""

import argparse
import os

import tomlkit  # type: ignore[import-not-found]

# Read the top-level version
with open("pyproject.toml", "r", encoding="utf-8") as f:
    top_data = tomlkit.load(f)

base_version = top_data["project"]["version"]

parser = argparse.ArgumentParser(description="Append a suffix to the base version")
parser.add_argument("suffix")
args = parser.parse_args()

version = base_version + args.suffix

projects = [
    ".",
    "deltakit-explorer",
    "deltakit-circuit",
    "deltakit-core",
    "deltakit-decode",
]

for project in projects:
    path = f"{project}/pyproject.toml"

    with open(path, "r", encoding="utf-8") as f:
        data = tomlkit.load(f)
        data["project"]["version"] = version

    with open(path, "w", encoding="utf-8") as f:
        tomlkit.dump(data, f)

filename = str(os.getenv("GITHUB_ENV"))
with open(filename, "a") as f:
    f.write(f"VERSION={version}\n")
