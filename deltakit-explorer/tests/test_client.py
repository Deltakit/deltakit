# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import importlib

from deltakit_explorer import __version__


def test_version():
    assert importlib.metadata.version("deltakit_explorer") == __version__
