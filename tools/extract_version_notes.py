"""
Extracts most recent release notes from `CHANGELOG.md` for use in a GitHub release
"""

import re
import sys
from pathlib import Path

if __name__ == "__main__":
    text = Path("CHANGELOG.md").read_text(encoding="utf-8")

    # Match headings starting with "## " (but not ###)
    pattern = re.compile(r"(^##\s+.*?$)", re.MULTILINE)
    matches = list(pattern.finditer(text))

    # Start of first section
    start = matches[0].end()

    # End is start of second heading (or EOF if only one heading)
    end = matches[1].start() if len(matches) > 1 else len(text)

    extracted_text = text[start:end].strip()
    Path(sys.argv[1]).write_text(extracted_text, encoding="utf-8")
