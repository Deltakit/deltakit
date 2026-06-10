# (c) Copyright Riverlane 2020-2025.
import pytest
import deltakit_stim as stim


@pytest.fixture
def empty_circuit():
    return stim.Circuit()
