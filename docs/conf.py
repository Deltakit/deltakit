# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
from __future__ import annotations

import os
import shutil

import deltakit

# -- Copy top-level files ------
# AFAICT, Sphinx can't see these unless we move them over
for filename in ["CONTRIBUTING.md", "CODE_OF_CONDUCT.md"]:
    old = os.path.join("..", filename)
    new = os.path.join(".", filename)
    # if not os.path.exists(new):
    shutil.copyfile(old, new)

# -- Project information -----------------------------------------------------

project = "Deltakit"
version = deltakit.__version__
copyright = "2020-2025, Riverlane"  # noqa: A001
author = "Riverlane Ltd and contributors"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_design",
]
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "substitution",
]

nb_execution_mode = "force"
nb_execution_excludepatterns = [
    "**/*.ipynb",
]
nb_execution_cache_path = ".jupyter_cache"
nb_execution_timeout = 300

autosectionlabel_prefix_document = True
autosummary_generate = True
autosummary_generate_overwrite = True  # Sphinx ≥ 7.2

templates_path = ["_templates"]
# include class docstring + __init__ docstring (shows Parameters)
autoclass_content = "both"
numpydoc_show_class_members = False
autodoc_member_order = "bysource"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    "json/*",
    "_build/jupyter_execute",
    "_build/html",
    "examples/notebooks/demo",
    "examples/notebooks/template_notebook.ipynb",
    "examples/notebooks/qmem/rotated_planar_quantum_memory_plots.ipynb",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Title for name link at the top of the documentation home page.
html_title = f"{project} v{version}"

# Favicon setup:
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_favicon
html_favicon = "logo/deltakit_favicon.png"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_permalinks_icon = "<span>#</span>"
html_theme = "sphinxawesome_theme"

# no link to an *.rst source
html_show_sourcelink = False

# Theme options are theme-specific and customise the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation at https://www.sphinx-doc.org/en/master/usage/theming.html.
html_theme_options = {
    "logo_light": "logo/deltakit_favicon.png",
    "logo_dark": "logo/deltakit_favicon.png",
}

# Configure MathJax v3
mathjax3_config = {
    "tex": {
        "packages": {"[+]": ["physics"]}  # Add the physics package
    },
    "loader": {
        "load": ["[tex]/physics"]  # Load physics extension
    },
}

highlight_language = "python3"

suppress_warnings = [
    "autosummary.import_cycle",
]
