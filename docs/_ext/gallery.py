"""Sphinx extension providing a ``gallery`` directive.

The directive renders a sphinx-design card grid followed by a hidden toctree
over the same documents. It writes the grid as reST or MyST source, matching
the page it is used in, and parses that in place, so it relies only on
sphinx-design's documented directives and roles.

Entries come from a YAML file beside the page (``gallery.yml`` by default): a
list of mappings, each with exactly one of ``doc`` (a docname relative to the
page) or ``ref`` (an autosectionlabel label, ``<docname>:<Section Title>``),
and optionally ``title``, ``summary``, ``level`` (``beginner``,
``intermediate`` or ``advanced``), ``tags`` (one tag or a list of them),
``meta`` and ``thumbnail`` (a path relative to the page, or to the source
directory when it starts with ``/``). Without that file, the sibling
``*.ipynb`` and ``*.md`` documents are listed in filename order.

All text is shown as written, never parsed as markup. A ``ref`` entry without
a ``title`` shows the section title from its ``ref``, so give it a ``title``
when the heading contains markup such as inline code or maths.

Entries without a ``thumbnail`` show the default Deltakit thumbnail.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import RSTState
from docutils.statemachine import StringList
from sphinx.util import docname_join, logging
from sphinx.util.docutils import SphinxDirective, switch_source_input

if TYPE_CHECKING:
    from sphinx.application import Sphinx

logger = logging.getLogger(__name__)

KEYS = {"doc", "ref", "title", "summary", "level", "tags", "meta", "thumbnail"}
# Keys that hold a name or a path, rather than text to show.
NAME_KEYS = {"doc", "ref", "thumbnail", "level"}
# Keys shown on a single line, as is each tag.
LINE_KEYS = {"title", "meta"}
LEVELS = ("beginner", "intermediate", "advanced")
DEFAULT_THUMBNAIL = "/_static/gallery/deltakit.svg"


def _value(key: str, value: Any, *, line: bool) -> str:
    """Return one value from the entries file as text.

    Args:
        key: The key the value belongs to.
        value: The value as YAML read it.
        line: Whether to join the value onto a single line.

    Returns:
        The value as a stripped string.

    Raises:
        ValueError: If the value is not text, or a number where text is shown.
    """
    # YAML reads an unquoted yes or no as a boolean.
    allowed = str if key in NAME_KEYS else (str, int, float)
    if isinstance(value, bool) or not isinstance(value, allowed):
        msg = f"{key} must be text, got {value!r} (quote it in the YAML)"
        raise ValueError(msg)
    # A folded or literal YAML scalar may hold line breaks.
    return " ".join(str(value).split()) if line else str(value).strip()


def _normalise(entry: Any, base: str) -> dict[str, Any]:
    """Validate one gallery entry and add the fields derived from it.

    Args:
        entry: One item of the entries list.
        base: Docname that relative ``doc`` entries are resolved against.

    Returns:
        A copy of the entry with its values as stripped strings, ``tags`` as a
        list of them, empty ones left out, and ``docname`` and ``section``
        added.

    Raises:
        ValueError: If the entry is not a mapping with exactly one of ``doc``
            and ``ref``, or has an unknown key, a value of the wrong type, or
            an unknown ``level`` or ``ref`` form.
    """
    if not isinstance(entry, dict) or not all(isinstance(key, str) for key in entry):
        msg = f"expected a mapping with text keys, got {entry!r}"
        raise ValueError(msg)
    if unknown := sorted(set(entry) - KEYS):
        msg = f"unknown keys {unknown}, expected some of {sorted(KEYS)}"
        raise ValueError(msg)
    values: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "tags":
            items = value if isinstance(value, list) else [value]
            tags = [_value(key, item, line=True) for item in items if item is not None]
            if tags := [tag for tag in tags if tag]:
                values[key] = tags
        elif value is not None and (text := _value(key, value, line=key in LINE_KEYS)):
            values[key] = text
    if ("doc" in values) == ("ref" in values):
        msg = f"needs exactly one of 'doc' or 'ref', got {entry!r}"
        raise ValueError(msg)
    if "level" in values:
        values["level"] = values["level"].lower()
        if values["level"] not in LEVELS:
            msg = f"level must be one of {LEVELS}, got {entry['level']!r}"
            raise ValueError(msg)
    if "doc" in values:
        docname, section = docname_join(base, values["doc"]), ""
    else:
        # Docnames cannot contain ":", but section titles can.
        docname, _, section = values["ref"].partition(":")
        if not docname or not section:
            msg = f"ref must be '<docname>:<Section Title>', got {values['ref']!r}"
            raise ValueError(msg)
    return {**values, "docname": docname, "section": section}


def _load(path: Path, base: str) -> list[dict[str, Any]]:
    """Read and validate a gallery entries file.

    Args:
        path: The YAML file.
        base: Docname that relative ``doc`` entries are resolved against.

    Returns:
        The normalised entries.

    Raises:
        ValueError: If the file does not hold a list of valid entries.
    """
    with path.open(encoding="utf-8") as stream:
        entries = yaml.safe_load(stream) or []
    if not isinstance(entries, list):
        msg = f"{path.name} must contain a list of entries"
        raise ValueError(msg)
    normalised = []
    for number, entry in enumerate(entries, start=1):
        try:
            normalised.append(_normalise(entry, base))
        except ValueError as err:
            msg = f"{path.name}, entry {number}: {err}"
            raise ValueError(msg) from err
    return normalised


def _text(line: str, *, rst: bool) -> str:
    """Escape one line so that it renders as plain text in reST and in MyST.

    Args:
        line: A line of text from the entries file.
        rst: Whether the line is for reST rather than MyST.

    Returns:
        The line with the characters that start inline markup escaped, and
        so is a character at its start that would make it a list, heading,
        quote or other block. Quotes, dashes and dots elsewhere are left for
        Sphinx's smart quotes, as in the rest of the page.
    """
    line = re.sub(r"([\\`*_\[\]<&$|:{}])", r"\\\1", line.strip())
    # reST also reads the bullets U+2022, U+2023 and U+2043, which CommonMark
    # cannot escape and does not treat as bullets.
    block = r"^([#+\-.=>(~\u2022\u2023\u2043])" if rst else r"^([#+\-.=>(~])"
    line = re.sub(block, r"\\\1", line)
    return re.sub(r"^(\w+)([.)])", r"\1\\\2", line)


def _uri(value: str) -> str:
    """Escape a value for an option that sphinx-design reads as a URI.

    Args:
        value: The link target or image path.

    Returns:
        The value with backslashes and whitespace escaped, which
        ``directives.uri`` would otherwise drop.
    """
    return re.sub(r"(\s)", r"\\\1", value.replace("\\", "\\\\"))


def _role(name: str, text: str, *, rst: bool) -> str:
    """Return an interpreted text role in reST or MyST syntax.

    Args:
        name: The role name.
        text: The role content, taken as written.
        rst: Whether to use reST syntax rather than MyST.

    Returns:
        The role markup.
    """
    if rst:
        return f":{name}:`" + re.sub(r"([\\`])", r"\\\1", text) + "`"
    # MyST takes the content as written, up to the next run of as many
    # backticks as opened it, so open with a longer run than the content has.
    ticks = "`" * (1 + max((len(run) for run in re.findall("`+", text)), default=0))
    if text.startswith("`") or text.endswith("`"):
        text = f" {text} "
    return f"{{{name}}}{ticks}{text}{ticks}"


def _directive(
    name: str,
    argument: str,
    options: list[tuple[str, str]],
    body: list[str],
    *,
    rst: bool,
) -> list[str]:
    """Return the source lines of a directive in reST or MyST syntax.

    Args:
        name: The directive name.
        argument: The directive argument, or an empty string.
        options: The directive options, as (name, value) pairs.
        body: The directive content lines.
        rst: Whether to use reST syntax rather than MyST.

    Returns:
        The directive markup, one line per item.
    """
    options_lines = [f":{key}: {value}".rstrip() for key, value in options]
    if rst:
        indented = [
            f"   {line}" if line else "" for line in [*options_lines, "", *body]
        ]
        return [f".. {name}:: {argument}".rstrip(), *indented]
    # A tilde fence, unlike a backtick one, may hold backticks in its argument.
    # It must be longer than any fence it contains.
    fence = "~" * max([3] + [len(line) - len(line.lstrip("~")) + 1 for line in body])
    return [f"{fence}{{{name}}} {argument}".rstrip(), *options_lines, "", *body, fence]


def _card(entry: dict[str, Any], *, rst: bool) -> list[str]:
    """Return the ``grid-item-card`` markup for one entry.

    Args:
        entry: A normalised entry.
        rst: Whether to use reST syntax rather than MyST.

    Returns:
        The card markup, one line per item.
    """

    def value(text: str) -> str:
        # MyST reads option values as YAML, where " #" would start a comment.
        return text if rst else "'" + text.replace("'", "''") + "'"

    if entry["section"]:
        # sphinx-design passes the target to std:ref as written, so it must
        # already be the normalised label.
        link = nodes.fully_normalize_name(f"{entry['docname']}:{entry['section']}")
        options = [("link", value(_uri(link))), ("link-type", "ref")]
    else:
        options = [("link", value(_uri(f"/{entry['docname']}"))), ("link-type", "doc")]
    image = entry.get("thumbnail", DEFAULT_THUMBNAIL)
    options += [
        ("img-top", value(_uri(image))),
        ("class-card", "dk-gallery-card"),
        ("class-img-top", "dk-gallery-thumb"),
    ]
    footer = []
    if "level" in entry:
        # A primary badge, so that the stylesheet can colour it by level.
        footer.append(_role("bdg-primary", entry["level"].capitalize(), rst=rst))
        options.append(("class-footer", f"dk-level-{entry['level']}"))
    footer += [_role("bdg", tag, rst=rst) for tag in entry.get("tags", [])]
    if "meta" in entry:
        footer.append(_role("literal", entry["meta"], rst=rst))
    body = [_text(line, rst=rst) for line in entry.get("summary", "").splitlines()]
    if footer:
        body += ["", "+++", " ".join(footer)]
    title = entry.get("title") or " ".join(entry["section"].split())
    if title:
        # The hidden link text is what screen readers announce, so match it.
        options.append(("link-alt", value(title)))
        title = _text(title, rst=rst)
    else:
        # The target may not have been read yet, so let Sphinx fill its title in.
        title = _role("doc", f"/{entry['docname']}", rst=rst)
    return _directive("grid-item-card", title, options, body, rst=rst)


class GalleryDirective(SphinxDirective):
    """Render a card grid for the entries of a YAML file, plus a hidden toctree.

    Attributes:
        option_spec: The ``entries`` file name, the grid ``columns``, and the
            ``toctree`` and ``no-toctree`` flags.
    """

    option_spec = {  # noqa: RUF012
        "entries": directives.unchanged_required,
        "columns": directives.unchanged,
        "toctree": directives.flag,
        "no-toctree": directives.flag,
    }

    def _siblings(self) -> list[dict[str, Any]]:
        """Return entries for the notebook and Markdown documents beside this page.

        Returns:
            One ``doc`` entry per sibling document, in filename order.
        """
        here = Path(self.env.srcdir, self.env.docname).parent
        stems = []
        for path in sorted(here.iterdir(), key=lambda path: path.name):
            docname = docname_join(self.env.docname, path.stem)
            # found_docs already leaves out anything matching exclude_patterns.
            if (
                path.suffix in {".ipynb", ".md"}
                and docname != self.env.docname
                and docname in self.env.found_docs
            ):
                stems.append(path.stem)
        # Sphinx cannot tell when siblings are added or removed, or when an
        # entries file appears, so read this page again on every build.
        self.env.note_reread()
        return [
            _normalise({"doc": stem}, self.env.docname) for stem in dict.fromkeys(stems)
        ]

    def run(self) -> list[nodes.Node]:
        """Parse the generated grid and toctree markup in place of the directive.

        Returns:
            The parsed nodes.
        """
        srcdir = Path(self.env.srcdir)
        path = (srcdir / self.env.docname).parent / self.options.get(
            "entries", "gallery.yml"
        )
        try:
            if path.is_file():
                self.env.note_dependency(str(path))
                entries = _load(path, self.env.docname)
            else:
                if "entries" in self.options:
                    logger.warning(
                        "gallery: %s not found, listing sibling documents instead",
                        path.name,
                        location=self.get_location(),
                    )
                entries = self._siblings()
        except (OSError, ValueError, yaml.YAMLError) as err:
            logger.warning("gallery: %s", err, location=self.get_location())
            return []
        if not entries:
            logger.warning(
                "gallery: %s",
                f"{path.name} has no entries"
                if path.is_file()
                else f"no {path.name} and no sibling .md or .ipynb documents",
                location=self.get_location(),
            )
            return []

        rst = isinstance(self.state, RSTState)
        cards = [line for entry in entries for line in [*_card(entry, rst=rst), ""]]
        columns = self.options.get("columns", "")
        lines = _directive("grid", columns, [("gutter", "3")], cards, rst=rst)
        if "no-toctree" not in self.options:
            # A gallery may link to sections of its own page, which a toctree
            # must not list.
            docnames = dict.fromkeys(
                f"/{entry['docname']}"
                for entry in entries
                if entry["docname"] != self.env.docname
            )
            options = [("hidden", ""), ("maxdepth", "1")]
            lines += ["", *_directive("toctree", "", options, list(docnames), rst=rst)]

        # Report problems in the generated markup at the directive's own line:
        # while parsing through the StringList items, and afterwards through
        # the location of each node.
        source, line = self.get_source_info()
        content = StringList(lines, items=[(source, line - 1)] * len(lines))
        container = nodes.Element()
        if rst:
            with switch_source_input(self.state, content):
                self.state.nested_parse(content, 0, container)
        else:
            self.state.nested_parse(content, 0, container)
        for node in container.findall(nodes.Element):
            node.source, node.line = source, line
        return container.children


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the ``gallery`` directive.

    Args:
        app: The Sphinx application.

    Returns:
        The extension metadata.
    """
    app.add_directive("gallery", GalleryDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
