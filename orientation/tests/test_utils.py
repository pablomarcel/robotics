import numpy as np
import math

from orientation.utils import normalize, skew, vex, vers, safe_acos, expm_so3, project_to_so3

def test_normalize_and_eps():
    u = np.array([3.0, 4.0, 0.0])
    un = normalize(u)
    assert np.isclose(np.linalg.norm(un), 1.0)
    z = np.zeros(3)
    assert np.allclose(normalize(z), z)  # stays zero

def test_skew_vex_roundtrip():
    v = np.array([0.2, -1.0, 0.7])
    S = skew(v)
    w = vex(S)
    assert np.allclose(v, w)

def test_vers_safe_acos():
    ang = 1.234
    assert np.isclose(vers(ang), 1 - math.cos(ang))
    assert 0.0 <= safe_acos(2.0) <= math.pi  # clamped

def test_expm_so3_small_angle_first_order():
    omega = np.array([1e-9, -2e-9, 3e-9])
    R = expm_so3(omega)
    assert np.allclose(R, np.eye(3) + skew(omega), atol=1e-12)

def test_expm_so3_matches_axis_angle():
    u = np.array([0.3, 0.4, -0.5]); u = u/np.linalg.norm(u)
    phi = 0.8
    R1 = expm_so3(u * phi)
    # Rodrigues via closed form
    c, s = np.cos(phi), np.sin(phi)
    U = skew(u)
    R2 = c*np.eye(3) + (1-c)*np.outer(u,u) + s*U
    assert np.allclose(R1, R2, atol=1e-9)

def test_project_to_so3():
    # Start with a true rotation_kinematics and perturb it
    u = np.array([0,0,1.0]); phi = 0.5
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    Rp = R + 1e-4*np.random.randn(3,3)
    Rproj = project_to_so3(Rp)
    should_be_I = Rproj.T @ Rproj
    assert np.allclose(should_be_I, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(Rproj), 1.0, atol=1e-8)
