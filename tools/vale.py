# Vale doesn't seem to have a built-in way to *ignore* directories
# This excludes `EXCLUDE_DIRS` by *including* all other directories
import subprocess
from pathlib import Path

EXCLUDE_DIRS = {".github", ".pixi", ".pytest_cache"}
INCLUDE_EXTS = {".md", ".rst", ".py"}

files = [
    str(p)
    for p in Path().rglob("*")
    if p.is_file()
    and p.suffix in INCLUDE_EXTS
    and not any(part in EXCLUDE_DIRS for part in p.parts)
]

if files:
    # check=False: we only need vale to print feedback to console; not raise an error
    subprocess.run(["vale", *files], check=False)
else:
    print("No files to lint.")  # noqa: T201
