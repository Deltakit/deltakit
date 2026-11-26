# (c) Copyright Riverlane 2020-2026.
"""Tests for the ``gallery`` Sphinx extension in ``docs/_ext``.

They build small projects to check what most often breaks when content is
added to a ``gallery.yml``. The extension needs the documentation dependencies,
which the ``test`` group does not install, so these tests are skipped without
them. Run them with ``just test-docs``.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
LOGO = DOCS / "logo" / "Deltakit_Blue_SecondaryLogo_RGB.png"

pytest.importorskip("sphinx_design")
pytest.importorskip("myst_parser")
sphinx_testing = pytest.importorskip("sphinx.testing.util")

CONF = f"""
import sys
sys.path.insert(0, {str(DOCS / "_ext")!r})
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
    "gallery",
]
autosectionlabel_prefix_document = True
exclude_patterns = ["_build"]
"""
ROOT = "Root\n====\n\n.. toctree::\n\n   gallery/index\n"
PAGES = {
    "gallery/alpha.md": "# Alpha\n\nText.\n",
    "gallery/beta.md": "# Beta\n\n## Step 1: setup\n\nText.\n",
}
# The gallery directive is on line 4 of the reST page.
RST_PAGE = "Gallery\n=======\n\n.. gallery::\n   :columns: 1 2 2 2\n"
MYST_PAGE = "# Gallery\n\n```{gallery}\n:columns: 1 2 2 2\n```\n"


@pytest.fixture
def build(tmp_path: Path) -> Iterator[Callable[[dict[str, str]], Any]]:
    """Build small Sphinx projects, and clean them up afterwards.

    Args:
        tmp_path: A directory for the project sources and builds.

    Yields:
        A function that builds a project from a mapping of file names to
        contents, and returns the Sphinx application.
    """
    apps = []

    def run(files: dict[str, str]):
        src = tmp_path / "src"
        for name, text in {
            "conf.py": CONF,
            "index.rst": ROOT,
            **PAGES,
            **files,
        }.items():
            (src / name).parent.mkdir(parents=True, exist_ok=True)
            (src / name).write_text(text, encoding="utf-8")
        (src / "logo").mkdir(exist_ok=True)
        shutil.copy(LOGO, src / "logo" / LOGO.name)
        app = sphinx_testing.SphinxTestApp(
            "html", srcdir=src, freshenv=True, status=StringIO(), warning=StringIO()
        )
        apps.append(app)
        app.build()
        return app

    yield run
    for app in apps:
        app.cleanup()


@pytest.mark.parametrize("page", ["index.rst", "index.md"])
def test_gallery_renders_its_entries(build, page: str) -> None:
    entries = """\
- doc: alpha
  title: "Alpha: *not* emphasis"
  level: advanced
  tags: [Cloud decoder]
- ref: "gallery/beta:Step 1: setup"
  level: beginner
"""
    source = RST_PAGE if page.endswith(".rst") else MYST_PAGE
    app = build({f"gallery/{page}": source, "gallery/gallery.yml": entries})

    assert app.warning.getvalue() == ""
    html = (app.outdir / "gallery" / "index.html").read_text(encoding="utf-8")
    assert html.count("dk-gallery-card") == len(PAGES)
    # Titles are shown as written, not parsed as markup.
    assert "Alpha: *not* emphasis" in html
    # A card links to a whole page, or to a section of one.
    assert 'href="alpha.html"' in html
    assert 'href="beta.html#step-1-setup"' in html
    assert "dk-level-advanced" in html
    assert "Cloud decoder" in html
    # Without a thumbnail, a card shows the Deltakit logo.
    assert (app.outdir / "_images" / LOGO.name).is_file()
    # The listed pages join the toctree, so they appear in the sidebar.
    assert app.env.toctree_includes["gallery/index"] == [
        "gallery/alpha",
        "gallery/beta",
    ]


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        ("- doc: alpha\n  description: typo\n", "unknown key 'description'"),
        ("- doc: alpha\n  level: expert\n", "level must be one of"),
        ("- doc: a\n  ref: gallery/beta:Beta\n", "needs exactly one of 'doc' or 'ref'"),
        ("- ref: gallery/beta\n", "ref must be '<docname>:<Section Title>'"),
        ("- doc: alpha\n  title: No\n", "title must be text"),
    ],
)
def test_gallery_warns_about_an_invalid_entry(build, entry: str, error: str) -> None:
    app = build({"gallery/index.rst": RST_PAGE, "gallery/gallery.yml": entry})
    assert (
        f"gallery/index.rst:4: WARNING: gallery: gallery.yml, entry 1: {error}"
        in app.warning.getvalue()
    )


@pytest.mark.parametrize(
    ("entry", "warning"),
    [
        (
            "- ref: 'gallery/beta:No such section'\n",
            "undefined label: 'gallery/beta:no such section'",
        ),
        ("- doc: gamma\n", "unknown document: '/gallery/gamma'"),
    ],
)
def test_gallery_warns_about_a_link_that_does_not_resolve(
    build, entry: str, warning: str
) -> None:
    app = build({"gallery/index.rst": RST_PAGE, "gallery/gallery.yml": entry})
    assert f"gallery/index.rst:4: WARNING: {warning}" in app.warning.getvalue()
