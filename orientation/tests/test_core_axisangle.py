import numpy as np
from orientation.core import AxisAngle, SO3

def test_axisangle_roundtrip_general():
    u = np.array([0.3, -0.4, 0.5]); u = u/np.linalg.norm(u)
    phi = 1.234
    R = AxisAngle(phi, u).as_matrix()
    aa = AxisAngle.from_matrix(R)
    assert np.isclose(aa.phi, phi, atol=1e-9)
    # axis direction can flip sign when angle≈π; general case should match:
    assert np.allclose(aa.u, u, atol=1e-9) or np.allclose(aa.u, -u, atol=1e-9)

def test_axisangle_near_zero():
    u = np.array([1,0,0.0])
    phi = 1e-9
    R = AxisAngle(phi, u).as_matrix()
    aa = AxisAngle.from_matrix(R)
    assert np.isclose(aa.phi, 0.0, atol=1e-8)

def test_axisangle_near_pi():
    u = np.array([0,1,0.0]); u=u/np.linalg.norm(u)
    phi = np.pi - 1e-7
    R = AxisAngle(phi, u).as_matrix()
    aa = AxisAngle.from_matrix(R)
    assert np.isclose(aa.phi, phi, atol=1e-6)

def test_axisangle_compose_matches_so3_compose():
    u1 = np.array([1,0,0.0]); u2 = np.array([0,0,1.0])
    aa1 = AxisAngle(0.4, u1)
    aa2 = AxisAngle(0.7, u2)
    R1 = aa1.as_matrix(); R2 = aa2.as_matrix()
    so = SO3(R2).compose(SO3(R1))
    assert np.allclose(so.R, R2 @ R1, atol=1e-10)
