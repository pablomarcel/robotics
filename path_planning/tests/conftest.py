import numpy as np
import pytest
from path_planning.app import PathPlannerApp
from path_planning.core import BoundaryConditions

@pytest.fixture
def app(): return PathPlannerApp()

@pytest.fixture
def unit_time(): return np.linspace(0,1,101)
