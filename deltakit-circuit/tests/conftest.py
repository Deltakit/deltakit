# (c) Copyright Riverlane 2020-2025.
import deltakit_stim
import pytest


@pytest.fixture
def empty_circuit():
    return deltakit_stim.Circuit()
