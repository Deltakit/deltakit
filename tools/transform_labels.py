"""
Transform labels to a JSON compatible list to be used in GitHub actions
with https://github.com/actions/labeler.
"""

import json
import os
from typing import Final

DEFAULT_PACKAGES: Final[frozenset[str]] = frozenset(
    [
        "deltakit",
        "deltakit-circuit",
        "deltakit-core",
        "deltakit-decode",
        "deltakit-explorer",
    ]
)


def transform(labels_str: str) -> str:
    labels = {label.strip() for label in labels_str.split(",")} & DEFAULT_PACKAGES
    return json.dumps(labels) if labels else json.dumps(DEFAULT_PACKAGES)


def main():
    all_labels = os.getenv("ALL_LABELS", "")

    with open(os.getenv("GITHUB_OUTPUT"), "a") as f:
        f.write(f"projects={transform(all_labels)}\n")


if __name__ == "__main__":
    main()
