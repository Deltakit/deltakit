# (c) Copyright Riverlane 2020-2026.
"""Tests for the ``gallery`` Sphinx extension in ``docs/_ext``.

They check how entries files and directive options are read, which is what
most often breaks when content is added to a gallery, without building any
documentation. The extension needs Sphinx and PyYAML, which the ``test`` group
does not install, so these tests are skipped without them. Run them with
``uv run --group docs --group test pytest tests/docs``.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
SOURCE_SUFFIXES = (".rst", ".md", ".ipynb")
GALLERIES = sorted(
    path for path in DOCS.rglob("gallery.yml") if "_build" not in path.parts
)

pytest.importorskip("sphinx")
pytest.importorskip("yaml")
sys.path.insert(0, str(DOCS / "_ext"))
gallery = importlib.import_module("gallery")


def load(tmp_path: Path, text: str) -> list[dict[str, Any]]:
    """Load the entries of a ``gallery.yml`` beside an ``examples/index`` page.

    Args:
        tmp_path: The directory to write the file in.
        text: The file contents.

    Returns:
        The normalised entries.
    """
    path = tmp_path / "gallery.yml"
    path.write_text(text, encoding="utf-8")
    return gallery._load(path, "examples/index")


@pytest.mark.parametrize(
    "path", GALLERIES, ids=lambda path: path.relative_to(DOCS).as_posix()
)
def test_gallery_lists_documents_and_thumbnails_that_exist(path: Path) -> None:
    page = path.relative_to(DOCS).with_name("index").as_posix()
    for entry in gallery._load(path, page):
        docname = entry["docname"]
        assert any(
            (DOCS / f"{docname}{suffix}").is_file() for suffix in SOURCE_SUFFIXES
        ), f"no document {docname!r}"
        thumbnail = entry.get(gallery.Key.THUMBNAIL, gallery.DEFAULT_THUMBNAIL)
        root = DOCS if thumbnail.startswith("/") else path.parent
        assert (root / thumbnail.lstrip("/")).is_file(), f"no image {thumbnail!r}"


@pytest.mark.parametrize("name", ["gallery.yml", ""], ids=["missing", "directory"])
def test_load_needs_an_existing_file(tmp_path: Path, name: str) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist or is not a file"):
        gallery._load(tmp_path / name, "examples/index")


def test_load_reads_each_key(tmp_path: Path) -> None:
    entries = load(
        tmp_path,
        """\
- doc: notebooks/alpha
  title: >-
    Alpha,
    folded
  level: Advanced
  tags: Cloud decoder
  summary: 42
  thumbnail:
  meta: ""
- ref: "examples/beta:Step 1: setup"
  tags: [Downloads data, null, ""]
""",
    )
    assert entries == [
        {
            "doc": "notebooks/alpha",
            "title": "Alpha, folded",
            "level": "advanced",
            "tags": ["Cloud decoder"],
            "summary": "42",
            "docname": "examples/notebooks/alpha",
            "section": "",
        },
        {
            "ref": "examples/beta:Step 1: setup",
            "tags": ["Downloads data"],
            "docname": "examples/beta",
            "section": "Step 1: setup",
        },
    ]


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("doc: alpha\n", "gallery.yml must contain a list of entries"),
        ("- doc: alpha\n- beta\n", "entry 2: expected a mapping with text keys"),
        ("- description: typo\n", "entry 1: unknown key 'description'"),
        ("- doc: alpha\n  level: expert\n", "entry 1: level must be one of"),
        ("- title: Alpha\n", "entry 1: needs exactly one of 'doc' or 'ref'"),
        (
            "- doc: alpha\n  ref: examples/beta:Beta\n",
            "entry 1: needs exactly one of 'doc' or 'ref'",
        ),
        ("- ref: examples/beta\n", "entry 1: ref must be '<docname>:<Section Title>'"),
        # YAML reads an unquoted No as false.
        ("- doc: alpha\n  title: No\n", "entry 1: title must be text"),
        ("- doc: 2024\n", "entry 1: doc must be text"),
    ],
)
def test_load_rejects_an_invalid_file(tmp_path: Path, text: str, error: str) -> None:
    with pytest.raises(ValueError, match=re.escape(error)):
        load(tmp_path, text)


def test_directive_parses_its_options() -> None:
    spec = gallery.GalleryDirective.option_spec
    assert spec["entries"]("tutorials.yml") == "tutorials.yml"
    assert spec["columns"]("1 2 2 3") == "1 2 2 3"
    assert spec["no-toctree"](None) is None
    with pytest.raises(ValueError, match="argument required"):
        spec["entries"](None)
    with pytest.raises(ValueError, match="no argument is allowed"):
        spec["no-toctree"]("yes")
