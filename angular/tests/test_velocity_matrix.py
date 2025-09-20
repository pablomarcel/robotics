import numpy as np
from angular.core import Transform

def test_velocity_matrix_matches_definition():
    R = np.eye(3); d = np.array([1.0, 2.0, 3.0])
    omega = np.array([0.1, 0.2, 0.3])
    Rdot = np.array([[0,-0.3,0.2],[0.3,0,-0.1],[-0.2,0.1,0]])  # skew(omega)
    ddot = np.array([0.5,-0.2,0.7])
    V = Transform(R, d).velocity_matrix(Rdot, ddot)
    assert np.allclose(V[:3,:3], Rdot)
    assert np.allclose(V[:3,3], ddot - Rdot @ d)
