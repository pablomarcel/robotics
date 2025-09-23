import numpy as np
from path.core import BoundaryConditions
from path.poly import CubicPoly
from path.segment import Piecewise1D

def test_piecewise_two_segments_continuity():
    s1 = CubicPoly(BoundaryConditions(0,0.5,0,1,0,0))
    s2 = CubicPoly(BoundaryConditions(0.5,1.0,1,2,0,0))
    pv = Piecewise1D(BoundaryConditions(0,1,0,2), segments=[s1,s2])
    # value continuity at 0.5
    assert np.isclose(pv.q(0.5), 1.0)
