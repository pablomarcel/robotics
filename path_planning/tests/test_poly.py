import numpy as np
import pytest
from path_planning.core import BoundaryConditions
from path_planning.poly import CubicPoly, QuinticPoly, SepticPoly

def test_cubic_matches_example_381_coeffs():
    # Example 381: q(0)=10, q(1)=45, qd(0)=qd(1)=0 -> a0=10, a1=0, a2=105, a3=-70
    bc = BoundaryConditions(0, 1, 10, 45, 0, 0)
    c = CubicPoly(bc)
    _ = c.q(0)  # trigger coefficient build
    a = c.coefficients()
    assert np.allclose(a, [10, 0, 105, -70], atol=1e-6)

def test_quintic_rest2rest_coeffs_example():
    bc = BoundaryConditions(0, 1, 10, 45, 0, 0, 0, 0)
    p = QuinticPoly(bc)
    _ = p.q(0)  # trigger coefficient build
    a = p.coefficients()
    assert len(a) == 6
    assert np.isclose(p.q(1), 45)

def test_septic_zero_jerk():
    """
    Septic is constructed to have zero jerk at both endpoints.
    Jerk is the first derivative of acceleration_kinematics (third derivative of position).
    We check with a symmetric first derivative of qdd, not a second derivative.
    """
    bc = BoundaryConditions(0, 1, 10, 45, 0, 0, 0, 0)
    p = SepticPoly(bc)

    dt = 1e-5  # small step for numerical differentiation

    def jerk(tt: float) -> float:
        # central difference for first derivative of acceleration_kinematics (jerk)
        return float((p.qdd(tt + dt) - p.qdd(tt - dt)) / (2 * dt))

    assert abs(jerk(0.0)) < 1e-5
    assert abs(jerk(1.0)) < 1e-5
