# velocity_kinematics/tests/test_analytic_mapping.py
"""
Analytic Jacobian mapping tests.

We verify that the implementation of the analytic Jacobian J_A agrees with the
independent construction JA = [ Jv ; G(φ)^{-1} Jw ], where φ are Euler angles
extracted from the current end-effector rotation_kinematics.

Two cases:
  1) ZYX (yaw-pitch-roll) for the planar 2R arm (non-singular: pitch=0).
  2) ZXZ for a 3R spherical wrist with a generic non-singular configuration.
"""

from __future__ import annotations

import numpy as np
import pytest

from velocity_kinematics import design


# ------------------------------ helpers ------------------------------------ #

def _euler_from_R_zyx(R: np.ndarray) -> tuple[bool, np.ndarray]:
    """
    Extract ZYX (yaw, pitch, roll) from rotation_kinematics matrix.
    Returns (ok, angles); ok=False indicates gimbal lock (pitch ≈ ±pi/2).
    """
    sy = -float(R[2, 0])  # -sin(pitch)
    if abs(sy) >= 1.0:
        pitch = np.sign(sy) * (np.pi / 2)
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
        return False, np.array([yaw, pitch, roll], dtype=float)
    pitch = np.arcsin(sy)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return True, np.array([yaw, pitch, roll], dtype=float)


def _G_zyx(angles: np.ndarray) -> np.ndarray:
    """
    Euler-rate map for ZYX such that ω = G(φ) φ̇, φ=[yaw(Z), pitch(Y), roll(X)].
    """
    yaw, pitch, roll = angles
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    # One common parametrization (matches velocity_kinematics.core implementation):
    G = np.array(
        [
            [0.0, -sr, cr * cp],
            [0.0,  cr, sr * cp],
            [1.0, 0.0,     -sp],
        ],
        dtype=float,
    )
    return G


def _euler_from_R_zxz(R: np.ndarray) -> tuple[bool, np.ndarray]:
    """
    Extract ZXZ angles (alpha, beta, gamma).
    Returns (ok, angles); ok=False when sin(beta) ≈ 0 (singular).
    """
    beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
    sb = np.sin(beta)
    if sb < 1e-9:
        alpha = 0.0
        gamma = np.arctan2(R[0, 1], R[0, 0])
        return False, np.array([alpha, beta, gamma], dtype=float)
    alpha = np.arctan2(R[0, 2], -R[1, 2])
    gamma = np.arctan2(R[2, 0], R[2, 1])
    return True, np.array([alpha, beta, gamma], dtype=float)


def _G_zxz(angles: np.ndarray) -> np.ndarray:
    """
    Euler-rate map for ZXZ such that ω = G(φ) φ̇, φ=[alpha(Z), beta(X), gamma(Z)].
    """
    alpha, beta, gamma = angles
    sb = np.sin(beta)
    cb = np.cos(beta)
    G = np.array(
        [
            [0.0, sb, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, cb, 0.0],
        ],
        dtype=float,
    )
    return G


# ------------------------------ tests -------------------------------------- #

def test_analytic_vs_geometric_mapping_zyx_planar2r():
    """
    For a planar 2R, orientation_kinematics is a pure yaw; pitch=0 (non-singular for ZYX).
    Check JA lower block equals G^{-1} Jw.
    """
    robot = design.planar_2r(1.0, 1.0)
    q = np.array([0.35, -0.22])
    Jg = robot.jacobian_geometric(q)
    JA = robot.jacobian_analytic(q, euler="ZYX")

    # Extract Euler angles from current EE rotation_kinematics
    R = robot.fk(q)["T_0e"][:3, :3]
    ok, ang = _euler_from_R_zyx(R)
    assert ok  # planar case has pitch=0, safely away from ZYX singularity

    G = _G_zyx(ang)
    Ginv = np.linalg.inv(G)

    # Split geometric J into translational and angular_velocity blocks
    Jv, Jw = Jg[:3, :], Jg[3:, :]
    JA_expected = np.vstack([Jv, Ginv @ Jw])

    assert JA.shape == Jg.shape
    assert np.allclose(JA, JA_expected, atol=1e-10)


def test_analytic_vs_geometric_mapping_zxz_wrist():
    """
    For a 3R spherical wrist, use ZXZ. Pick a generic non-singular q (beta not ~0 or pi).
    """
    wrist = design.spherical_wrist(wrist_type=1, d_tool=0.0)  # type 1 ~ Z–X–Z
    # Choose a configuration away from ZXZ singularities:
    # beta comes from R[2,2] = cos(beta), so avoid cos(beta) ~ ±1
    q = np.array([0.6, -0.9, 0.7])
    Jg = wrist.jacobian_geometric(q)
    JA = wrist.jacobian_analytic(q, euler="ZXZ")

    R = wrist.fk(q)["T_0e"][:3, :3]
    ok, ang = _euler_from_R_zxz(R)
    # If unlucky (numerically near singular), tweak q slightly and retry once
    if not ok:
        q = q + np.array([0.03, -0.02, 0.01])
        Jg = wrist.jacobian_geometric(q)
        JA = wrist.jacobian_analytic(q, euler="ZXZ")
        R = wrist.fk(q)["T_0e"][:3, :3]
        ok, ang = _euler_from_R_zxz(R)
    assert ok, "ZXZ extraction hit a singularity; try a different configuration."

    G = _G_zxz(ang)
    Ginv = np.linalg.inv(G)
    Jv, Jw = Jg[:3, :], Jg[3:, :]
    JA_expected = np.vstack([Jv, Ginv @ Jw])

    assert JA.shape == Jg.shape
    assert np.allclose(JA, JA_expected, atol=1e-9)
