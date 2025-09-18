import numpy as np
from orientation.core import SO3

def test_group_ops_and_conversions():
    a1 = SO3.from_axis_angle(0.2, [1,0,0])
    a2 = SO3.from_axis_angle(0.3, [0,1,0])
    c = a2.compose(a1)
    inv = c.inverse()
    I = c.compose(inv).R
    assert np.allclose(I, np.eye(3), atol=1e-10)
    # conversions
    q = c.to_quaternion()
    rv = c.to_rodrigues()
    aa = c.to_axis_angle()
    assert np.allclose(SO3.from_quaternion(q).R, c.R, atol=1e-10)
    assert np.allclose(SO3.from_rodrigues(rv.w).R, c.R, atol=1e-10)
    # properties
    assert 0.0 <= c.angle() <= np.pi
    assert np.isclose(np.linalg.norm(c.axis()), 1.0, atol=1e-9)

def test_orthogonality_det():
    s = SO3.from_axis_angle(0.77, [0,0,1])
    RtR = s.R.T @ s.R
    assert np.allclose(RtR, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(s.R), 1.0, atol=1e-10)
