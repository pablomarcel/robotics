import numpy as np
import pytest
from control_techniques.app import ControlApp

@pytest.fixture
def app(): return ControlApp()

@pytest.fixture
def near():
    def _near(a,b,eps=1e-6): assert np.allclose(a,b,atol=eps,rtol=0)
    return _near
