import numpy as np
import pytest
from robot_dynamics.design import DHChainBuilder

@pytest.fixture
def planar2r_model():
    model, _ = DHChainBuilder.planar_2r(1.0, 1.0, 1.0, 1.0)
    return model

@pytest.fixture
def sample_state():
    return (np.array([0.2, -0.3]), np.array([0.1, 0.05]), np.array([0.0, 0.0]))