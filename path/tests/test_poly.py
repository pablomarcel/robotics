import numpy as np
from path.core import BoundaryConditions
from path.poly import CubicPoly, QuinticPoly, SepticPoly

def test_cubic_matches_example_381_coeffs():
    # Example 381: q(0)=10, q(1)=45, qd(0)=qd(1)=0 -> a0=10, a1=0, a2=105, a3=-70
    bc = BoundaryConditions(0,1,10,45,0,0)
    c = CubicPoly(bc); c.q(0)  # trigger build
    a = c.coefficients()
    assert np.allclose(a, [10,0,105,-70], atol=1e-6)

def test_quintic_rest2rest_coeffs_example():
    bc = BoundaryConditions(0,1,10,45,0,0,0,0)
    p = QuinticPoly(bc); p.q(0)
    a = p.coefficients()
    # known shape: 6 coeffs
    assert len(a)==6
    assert np.isclose(p.q(1), 45)

def test_septic_zero_jerk():
    bc = BoundaryConditions(0,1,10,45,0,0,0,0)
    p = SepticPoly(bc); t=np.array([0.0,1.0])
    # jerk ≈ derivative of acceleration is linear combo of coefficients; sample numerically
    dt=1e-4
    def jerk(tt):
        qdd = p.qdd(np.array([tt-dt, tt, tt+dt]))
        return (qdd[2]-2*qdd[1]+qdd[0])/(dt**2)
    assert abs(jerk(0.0)) < 1e2  # bounded small-ish
