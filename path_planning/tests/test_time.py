import numpy as np
from path_planning.core import BoundaryConditions
from path_planning.time import LSPB, QuinticTime

def test_quintic_zero_vel_acc():
    bc = BoundaryConditions(0,1,10,45)
    q = QuinticTime(bc)
    assert np.isclose(q.qd(0), 0)
    assert np.isclose(q.qdd(1), 0)
    assert np.isclose(q.q(0), 10)
    assert np.isclose(q.q(1), 45)

def test_lspb_shapes():
    bc = BoundaryConditions(0,1,0,1)
    tr = LSPB(bc, vmax=1.5)
    t = np.linspace(0,1,50)
    assert tr.q(t).shape == t.shape
    assert tr.qd(t).shape == t.shape
