import numpy as np
from control.core import MassSpringDamper

def test_msd_matrices_shapes():
    msd = MassSpringDamper(1.0, 0.8, 10.0)
    assert msd.A().shape == (2,2)
    assert msd.B().shape == (2,1)
    assert msd.C().shape == (1,2)
    assert msd.D().shape == (1,1)
