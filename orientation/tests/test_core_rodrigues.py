import numpy as np
from orientation.core import AxisAngle, Quaternion, SO3

def test_quaternion_matrix_equivalence():
    u = np.array([0.3,-0.4,0.2]); u = u/np.linalg.norm(u)
    phi = 1.1
    R1 = AxisAngle(phi, u).as_matrix()
    R2 = Quaternion.from_axis_angle(phi, u).as_matrix()
    assert np.allclose(R1, R2, atol=1e-10)

def test_quaternion_from_matrix_roundtrip():
    u = np.array([0,0,1.0]); phi = 0.75
    R = AxisAngle(phi, u).as_matrix()
    q = Quaternion.from_matrix(R)
    Rb = q.as_matrix()
    assert np.allclose(R, Rb, atol=1e-10)

def test_quaternion_multiply_equals_composition():
    u1 = np.array([1,0,0.0]); u2 = np.array([0,1,0.0])
    q1 = Quaternion.from_axis_angle(0.2, u1)
    q2 = Quaternion.from_axis_angle(0.5, u2)
    q = q2.multiply(q1)
    Rq = q.as_matrix()
    Rs = SO3.from_axis_angle(0.5, u2).compose(SO3.from_axis_angle(0.2, u1)).R
    assert np.allclose(Rq, Rs, atol=1e-10)

def test_quaternion_rotate_vector():
    u = np.array([0,0,1.0]); phi = np.pi/4
    q = Quaternion.from_axis_angle(phi, u)
    v = np.array([1,0,0.0])
    r = q.rotate(v)
    R = q.as_matrix()
    assert np.allclose(r, R @ v, atol=1e-10)
