import numpy as np
import pytest


@pytest.fixture
def random_generator():
    return np.random.default_rng()
