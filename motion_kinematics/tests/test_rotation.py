# motion_kinematics/tests/test_rotation.py
import math
import numpy as np
import pytest

from motion_kinematics.core import Rotation
from motion_kinematics.utils import is_rotation_matrix


def test_rz_90_matches_axis_angle():
    Rz = Rotation.Rz(math.pi / 2).as_matrix()
    Ra = Rotation.from_axis_angle([0, 0, 1], math.pi / 2).as_matrix()
    assert np.allclose(Rz, Ra, atol=1e-12)
    assert is_rotation_matrix(Rz)


def test_rx_ry_rz_are_proper_rotations():
    for factory, ang in [(Rotation.Rx, 0.1), (Rotation.Ry, -0.7), (Rotation.Rz, 1.2)]:
        R = factory(ang).as_matrix()
        assert is_rotation_matrix(R), f"Not a proper rotation for {factory.__name__}"


def test_inverse_is_transpose_and_composition_identity():
    R = Rotation.from_axis_angle([1, 2, 3], 0.7).as_matrix()
    Rt = Rotation(R.T).as_matrix()
    I1 = R @ Rt
    I2 = Rt @ R
    assert np.allclose(I1, np.eye(3), atol=1e-12)
    assert np.allclose(I2, np.eye(3), atol=1e-12)


def test_same_axis_composition_adds_angles():
    axis = np.array([1.0, -2.0, 0.5])
    a, b = 0.25, -0.4
    R1 = Rotation.from_axis_angle(axis, a).as_matrix()
    R2 = Rotation.from_axis_angle(axis, b).as_matrix()
    R_sum = Rotation.from_axis_angle(axis, a + b).as_matrix()
    assert np.allclose(R1 @ R2, R_sum, atol=1e-12)


def test_zero_angle_is_identity_any_axis():
    axis = [3, -5, 7]  # not unit on purpose
    R = Rotation.from_axis_angle(axis, 0.0).as_matrix()
    assert np.allclose(R, np.eye(3), atol=1e-12)


def test_small_angle_first_order_term():
    # For small phi, R ≈ I + phi [u]_x
    axis = np.array([0.2, 0.3, -0.5])
    phi = 1e-6
    R = Rotation.from_axis_angle(axis, phi).as_matrix()
    # Build I + phi [u]_x
    axis_u = axis / np.linalg.norm(axis)
    x, y, z = axis_u
    K = np.array([[0, -z,  y],
                  [z,  0, -x],
                  [-y, x,  0]], dtype=float)
    approx = np.eye(3) + phi * K
    assert np.allclose(R, approx, atol=1e-12)


def test_random_axes_yield_orthonormal_mats():
    rng = np.random.default_rng(123)
    for _ in range(10):
        a = rng.standard_normal(3)
        ang = rng.uniform(-math.pi, math.pi)
        R = Rotation.from_axis_angle(a, ang).as_matrix()
        assert is_rotation_matrix(R)
        # Columns are orthonormal
        cols = [R[:, i] for i in range(3)]
        for i in range(3):
            assert np.isclose(np.linalg.norm(cols[i]), 1.0, atol=1e-12)
        for i in range(3):
            for j in range(i + 1, 3):
                assert np.isclose(cols[i] @ cols[j], 0.0, atol=1e-12)
