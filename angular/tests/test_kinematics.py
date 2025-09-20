import numpy as np
from angular.core import KinematicsEngine

def test_rigid_body_point_velocity():
    omega = np.array([0,0,2.0])
    dB = np.array([1.0, 0.0, 0.0]); dBdot = np.array([0.0, 1.0, 0.0])
    rP = np.array([1.0, 2.0, 0.0])
    v = KinematicsEngine.rigid_body_point_velocity(omega, rP, dB, dBdot)
    # v = ω × (rP-dB) + dBdot = [0,0,2] × [0,2,0] + [0,1,0] = [-4,0,0] + [0,1,0]
    assert np.allclose(v, np.array([-4.0, 1.0, 0.0]))
