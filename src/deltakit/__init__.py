# (c) Copyright Riverlane 2020-2025.
import importlib.metadata

__version__: str = importlib.metadata.version(distribution_name=__package__)

from deltakit_explorer import Client

# Prevent import of `importlib` (and any other non-public objects) from this module.
del importlib

# List only public members in `__all__`.
__all__: list[str] = ["Client", "compile", "visualise"]
