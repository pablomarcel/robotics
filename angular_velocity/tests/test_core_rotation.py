import numpy as np
from angular_velocity.core import Rotation

def test_omega_from_Rdot_and_back():
    phi = 0.2
    u = np.array([0,0,1.0])
    R = Rotation.from_axis_angle(u, phi).R
    omega = np.array([0,0,0.7])
    Rdot = Rotation(R).Rdot_from_omega(omega)
    Omega = Rotation(R).omega_tilde_from_Rdot(Rdot)
    assert np.allclose(Omega, np.array([[0,-0.7,0],[0.7,0,0],[0,0,0]]), atol=1e-9)
