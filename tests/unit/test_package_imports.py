# (c) Copyright Riverlane 2020-2026.
# Import tests require a fresh interpreter.
import subprocess  # nosec B404
import sys
import textwrap

import pytest


def _run_in_fresh_interpreter(script):
    # Only fixed test scripts are executed.
    subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "statement",
    ["import deltakit", "import deltakit.circuit", "import deltakit.core"],
)
def test_import_does_not_load_explorer(statement):
    script = f"""
import sys
{statement}
import deltakit

assert deltakit.__version__
assert "Client" in dir(deltakit)
assert not hasattr(deltakit, "missing_attribute")
assert not any(
    name == "deltakit_explorer" or name.startswith("deltakit_explorer.")
    for name in sys.modules
)
"""
    _run_in_fresh_interpreter(script)


@pytest.mark.parametrize(
    "statement",
    [
        "from deltakit import Client",
        "from deltakit import *",
        "Client = deltakit.Client",
    ],
)
def test_client_import_preserves_public_api(statement):
    script = f"""
import deltakit
{statement}
from deltakit_explorer import Client as ExplorerClient

assert Client is ExplorerClient
assert deltakit.Client is Client
assert vars(deltakit)["Client"] is Client
assert deltakit.__all__ == ["Client"]
"""
    _run_in_fresh_interpreter(script)


def test_unknown_attribute_raises_attribute_error():
    script = textwrap.dedent("""
        import deltakit

        try:
            deltakit.missing_attribute
        except AttributeError as error:
            expected = "module 'deltakit' has no attribute 'missing_attribute'"
            assert str(error) == expected
        else:
            raise AssertionError("Expected AttributeError")
    """)
    _run_in_fresh_interpreter(script)
