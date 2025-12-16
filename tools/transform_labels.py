"""
Transform labels to a JSON compatible list to be used in GitHub actions
with https://github.com/actions/labeler.
"""

import json
import os
from typing import Final

DEFAULT_PACKAGES: Final[dict[str, str]] = {
    "deltakit": "./",
    "deltakit-circuit": "./deltakit-circuit",
    "deltakit-core": "./deltakit-core",
    "deltakit-decode": "./deltakit-decode",
    "deltakit-explorer": "./deltakit-explorer",
}


def filter_labels(labels_str: str) -> str:
    present_labels = sorted(
        {label.strip() for label in labels_str.split(",")} & DEFAULT_PACKAGES.keys()
    )
    if not present_labels:
        present_labels = sorted(DEFAULT_PACKAGES)
    matrix = [
        {"project": project, "path": DEFAULT_PACKAGES[project]}
        for project in present_labels
    ]
    return json.dumps(matrix)


def main():
    all_labels = os.getenv("ALL_LABELS", "")

    github_output_path = os.getenv("GITHUB_OUTPUT")
    if github_output_path is None:
        msg = "The environment variable GITHUB_OUTPUT should be set."
        raise ValueError(msg)

    with open(github_output_path, "a") as f:
        f.write(f"projects={filter_labels(all_labels)}\n")


if __name__ == "__main__":
    main()
