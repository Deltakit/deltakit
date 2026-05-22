# (c) Copyright Riverlane 2020-2025.
import deltakit_stim as stim
import pytest


@pytest.fixture
def empty_circuit():
    return stim.Circuit()
