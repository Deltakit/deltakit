# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING as _TYPE_CHECKING

__version__: str = importlib.metadata.version(distribution_name=__package__)

if _TYPE_CHECKING:
    from deltakit_explorer import Client as Client

# Prevent import of `importlib` (and any other non-public objects) from this module.
del importlib

# List only public members in `__all__`.
__all__: list[str] = ["Client"]


def __getattr__(name: str) -> type[Client]:
    """Load the cloud client only when it is requested.

    Args:
        name: Attribute to retrieve.

    Returns:
        The Explorer client class.

    Raises:
        AttributeError: If the attribute is not a public export.
    """
    if name == "Client":
        from deltakit_explorer import Client  # noqa: PLC0415 - deferred public export

        globals()[name] = Client
        return Client

    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


def __dir__() -> list[str]:
    """List module attributes, including public exports that are not loaded yet.

    Returns:
        The sorted attribute names.
    """
    return sorted(set(globals()) | set(__all__))
