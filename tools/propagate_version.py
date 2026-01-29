"""
This module is used by the "release" pixi task in order to propagate the version in the
toml file of the top-level project into the toml files of the sub-projects in the
monorepo.
"""

import tomlkit  # type: ignore[import-not-found]

# Read the top-level version
with open("pyproject.toml", "r", encoding="utf-8") as f:
    top_data = tomlkit.load(f)

version = top_data["project"]["version"]

projects = ["deltakit-explorer", "deltakit-circuit", "deltakit-core", "deltakit-decode"]

for project in projects:
    path = f"{project}/pyproject.toml"

    with open(path, "r", encoding="utf-8") as f:
        data = tomlkit.load(f)
        data["project"]["version"] = version

    with open(path, "w", encoding="utf-8") as f:
        tomlkit.dump(data, f)
