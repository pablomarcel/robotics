# inverse/tests/test_utils_so3.py
# Focused tests for SO(3)/SE(3) helpers used by the IK solvers.

from __future__ import annotations

import math
import numpy as np
import pytest

from inverse import utils as U


def _exp_so3(w: np.ndarray) -> np.ndarray:
    """
    Minimal Rodrigues' formula for tests: exp([w]^).
    """
    w = np.asarray(w, float).reshape(3)
    th = np.linalg.norm(w)
    if th < 1e-14:
        return np.eye(3)
    k = w / th
    K = U.skew(k)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


@pytest.mark.parametrize(
    "axis",
    [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])],
)
def test_so3_log_matches_small_rotations_on_axes(axis: np.ndarray):
    """
    For tiny rotations about canonical axes, log(exp(w)) ≈ w.
    """
    w = axis / np.linalg.norm(axis) * 1e-7
    R = _exp_so3(w)
    w_log = U.so3_log(R)
    assert np.allclose(w, w_log, atol=1e-9)


def test_so3_log_matches_random_small_rotations():
    """
    For random tiny rotations, the log map should recover the rotation_kinematics vector.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.normal(size=3)
        w = w / (np.linalg.norm(w) + 1e-12) * 5e-7  # tiny angle
        R = _exp_so3(w)
        w_log = U.so3_log(R)
        assert np.allclose(w, w_log, atol=1e-8)


def test_pose_error_orientation_small_vs_so3():
    """
    pose_error(..., mode="small") should agree with mode="so3" for small angles.
    """
    th = 1e-6
    w = np.array([0.3, -0.7, 0.1], dtype=float)
    w = w / np.linalg.norm(w) * th
    Rerr = _exp_so3(w)
    T_curr = np.eye(4)
    T_des = np.eye(4)
    T_des[:3, :3] = Rerr

    e_small = U.pose_error(T_curr, T_des, mode="small")
    e_so3 = U.pose_error(T_curr, T_des, mode="so3")

    # Pure orientation_kinematics error; position components zero
    assert np.allclose(e_small[:3], 0.0, atol=1e-12)
    assert np.allclose(e_so3[:3], 0.0, atol=1e-12)
    # Orientation components nearly equal for small angles
    assert np.allclose(e_small[3:], e_so3[3:], atol=1e-9)


@pytest.mark.parametrize(
    "rpy",
    [
        (0.0, 0.0, 0.0),
        (0.4, -1.1, 2.2),
        (-0.7, 0.8, -2.6),
        (1.2, -1.55, 0.3),  # near gimbal (pitch≈-π/2)
    ],
)
def test_rpy_roundtrip(rpy):
    """
    R_to_rpy(rpy_to_R(.)) should recover the original angles (up to periodicity).
    Loosen tolerance near gimbal singularities.
    """
    roll, pitch, yaw = rpy
    R = U.rpy_to_R(roll, pitch, yaw)
    r2, p2, y2 = U.R_to_rpy(R)

    # Normalize differences to (-π, π]
    def ang_diff(a, b):
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))

    tol = 5e-6 if abs(abs(pitch) - math.pi / 2) < 0.03 else 5e-7
    assert abs(ang_diff(roll, r2)) < tol
    assert abs(ang_diff(pitch, p2)) < tol
    assert abs(ang_diff(yaw, y2)) < tol
