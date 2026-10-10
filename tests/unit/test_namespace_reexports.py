# (c) Copyright Riverlane 2020-2026.
import importlib
import pkgutil

import pytest

import deltakit

CLOUD_DECODERS = {
    "ACDecoder",
    "BeliefMatchingDecoder",
    "BPOSDecoder",
    "CCDecoder",
    "LCDecoder",
    "MWPMDecoder",
}

NAMESPACE_MODULES = sorted(
    module.name
    for module in pkgutil.walk_packages(deltakit.__path__, prefix="deltakit.")
)


def _public_names(module):
    return {
        name
        for name in getattr(module, "__all__", dir(module))
        if not name.startswith("_")
    }


@pytest.mark.parametrize("module_name", NAMESPACE_MODULES)
def test_namespace_mirrors_upstream_public_api(module_name):
    module = importlib.import_module(module_name)
    upstream = importlib.import_module(module_name.replace("deltakit.", "deltakit_", 1))

    exported = set(module.__all__)
    expected = _public_names(upstream)
    if module_name == "deltakit.decode":
        expected |= CLOUD_DECODERS

    assert expected - exported == set(), "public names missing from re-export"
    assert exported - expected == set(), "unexpected names leaked into __all__"
