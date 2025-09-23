import numpy as np
import pytest
from path.app import PathPlannerApp
from path.core import BoundaryConditions

@pytest.fixture
def app(): return PathPlannerApp()

@pytest.fixture
def unit_time(): return np.linspace(0,1,101)
