# motion/tests/test_se3_inverse.py
import math
import numpy as np
import pytest

from motion.core import SE3, Rotation
from motion.utils import is_se3, is_rotation_matrix


def test_inverse_rule_Tinv_T_is_identity():
    R = Rotation.Rz(0.3).as_matrix()
    t = np.array([1.0, -2.0, 3.0])
    T = SE3(R, t)
    I1 = (T.inv() @ T).as_matrix()
    I2 = (T @ T.inv()).as_matrix()
    assert np.allclose(I1, np.eye(4), atol=1e-12)
    assert np.allclose(I2, np.eye(4), atol=1e-12)
    assert is_se3(T.as_matrix())


def test_inverse_blocks_match_formula():
    R = Rotation.from_axis_angle([1, 2, 3], 0.7).as_matrix()
    t = np.array([0.5, -0.25, 0.75])
    T = SE3(R, t)
    Ti = T.inv().as_matrix()

    # Expected inverse: [ R^T  -R^T t ; 0 0 0 1 ]
    RT = R.T
    expected = np.eye(4)
    expected[:3, :3] = RT
    expected[:3, 3] = -RT @ t

    assert np.allclose(Ti, expected, atol=1e-12)
    assert is_rotation_matrix(Ti[:3, :3])
    assert is_se3(Ti)


def test_composition_matches_block_form():
    A = SE3(Rotation.Rx(0.2).as_matrix(), np.array([0.1, 0.0, 0.0]))
    B = SE3(Rotation.Ry(-0.4).as_matrix(), np.array([0.0, 0.2, 0.0]))
    C = A @ B
    T = C.as_matrix()

    R_expected = A.R @ B.R
    t_expected = A.R @ B.t + A.t

    assert np.allclose(T[:3, :3], R_expected, atol=1e-12)
    assert np.allclose(T[:3, 3], t_expected, atol=1e-12)
    assert is_rotation_matrix(T[:3, :3])
    assert is_se3(T)


def test_associativity_of_composition():
    A = SE3(Rotation.Rx(0.1).as_matrix(), np.array([0.1, 0.2, 0.3]))
    B = SE3(Rotation.Ry(0.2).as_matrix(), np.array([-0.3, 0.0, 0.1]))
    C = SE3(Rotation.Rz(-0.3).as_matrix(), np.array([0.0, -0.2, 0.2]))
    left = (A @ B) @ C
    right = A @ (B @ C)
    assert np.allclose(left.as_matrix(), right.as_matrix(), atol=1e-12)


def test_inverse_of_product_equals_reverse_product_of_inverses():
    A = SE3(Rotation.Rz(0.4).as_matrix(), np.array([1.0, 0.0, 0.0]))
    B = SE3(Rotation.Rx(-0.7).as_matrix(), np.array([0.0, 1.0, 0.0]))
    C = A @ B
    lhs = C.inv().as_matrix()
    rhs = (B.inv() @ A.inv()).as_matrix()
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_apply_point_and_apply_points():
    R = Rotation.Rz(math.pi / 2).as_matrix()
    t = np.array([1.0, 2.0, 3.0])
    T = SE3(R, t)

    p = np.array([1.0, 0.0, 0.0])
    q = T.apply(p)  # rotate x→y, then translate
    assert np.allclose(q, np.array([1.0, 3.0, 3.0]), atol=1e-12)

    P = np.array([[1.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, -1.0]])
    Q = T.apply_points(P)
    # Check first row matches single-point result
    assert np.allclose(Q[0], q, atol=1e-12)
    # Spot check the others
    assert np.allclose(Q[1], np.array([-1.0, 2.0, 3.0]), atol=1e-12)
    assert np.allclose(Q[2], np.array([1.0, 2.0, 2.0]), atol=1e-12)


def test_from_matrix_roundtrip():
    R = Rotation.from_axis_angle([0.3, -0.7, 0.2], 1.1).as_matrix()
    t = np.array([-0.5, 0.25, 0.75])
    T0 = SE3(R, t)
    T = SE3.from_matrix(T0.as_matrix())
    assert np.allclose(T.R, R, atol=1e-12)
    assert np.allclose(T.t, t, atol=1e-12)
    assert is_se3(T.as_matrix())
