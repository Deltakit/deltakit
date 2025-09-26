"""
Transform labels to JSON compatible list to be used in GitHub actions with https://github.com/actions/labeler
"""
import json
import os

DEFAULT_PACKAGES = ["deltakit-explorer", "deltakit-circuit", "deltakit-core", "deltakit-decode"]


def transform(labels_str: str) -> str:
    if not labels_str.strip():
        return json.dumps(DEFAULT_PACKAGES)
    labels = [label.strip() for label in labels_str.split(",") if label.strip() in DEFAULT_PACKAGES]
    if not labels:
        return json.dumps(DEFAULT_PACKAGES)
    return json.dumps(labels)

def main():
    all_labels = os.getenv("ALL_LABELS", "")

    with open(os.getenv("GITHUB_OUTPUT"), "a") as f:
        f.write(f"JSON_LABELS_ALL={transform(all_labels)}\n")

if __name__ == "__main__":
    main()
