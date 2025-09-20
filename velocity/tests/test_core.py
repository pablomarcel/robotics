# velocity/tests/test_core.py
"""
Unit tests for Velocity Kinematics core.

Covers:
- FK & geometric Jacobian for planar 2R (matches textbook)
- Jacobian columns helper (design.jacobian_columns)
- Spherical-wrist zero-block check (with wrist center as TCP)
- Resolved-rates inverse velocity (least-squares consistency)
- Newton–Raphson IK (position-only) convergence
- Analytic Jacobian basic shape / fallback-safety
"""

from __future__ import annotations

import numpy as np
import pytest

from velocity import core, design


def _deg(*vals):
    return np.deg2rad(np.array(vals, dtype=float))


# --------------------------------------------------------------------------- #
# Planar 2R – textbook Jacobian
# --------------------------------------------------------------------------- #

def test_planar2r_jacobian_matches_textbook():
    robot = design.planar_2r(l1=1.0, l2=1.0)
    th1, th2 = _deg(30, -20)
    q = np.array([th1, th2], dtype=float)

    J = robot.jacobian_geometric(q)

    # Textbook planar 2R Jacobian (XY planar velocity + about-Z angular)
    l1 = l2 = 1.0
    s1, c1 = np.sin(th1), np.cos(th1)
    s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
    J_expected = np.array([
        [-l1 * s1 - l2 * s12, -l2 * s12],
        [ l1 * c1 + l2 * c12,  l2 * c12],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
    ])

    assert J.shape == (6, 2)
    assert np.allclose(J, J_expected, atol=1e-10)


def test_fk_frames_and_shapes():
    robot = design.planar_2r(1.0, 1.0)
    q = np.array([0.1, 0.2])
    out = robot.fk(q)
    T_0e = out["T_0e"]
    frames = out["frames"]
    assert T_0e.shape == (4, 4)
    # frames includes base and final TCP: length = n+1
    assert isinstance(frames, list)
    assert len(frames) == 1 + 2
    assert all(F.shape == (4, 4) for F in frames)


def test_design_jacobian_columns_agree_with_J():
    robot = design.planar_2r(0.8, 0.6)
    q = np.array([0.3, -0.4])
    cols = design.jacobian_columns(robot, q)
    J = robot.jacobian_geometric(q)
    assert len(cols) == J.shape[1]
    for i, c in enumerate(cols):
        assert np.allclose(c.reshape(-1), J[:, i], atol=1e-12)


# --------------------------------------------------------------------------- #
# Spherical wrist – zero upper-right block (at wrist center)
# --------------------------------------------------------------------------- #

def test_spherical_wrist_zero_block_at_center():
    # Build a 6R wrist-decoupled arm where TCP is at the wrist center (d_tool=0)
    robot = design.six_dof_spherical(l1=0.5, l2=0.4, wrist_type=1, d_tool=0.0)
    q = np.array([0.2, -0.3, 0.1, -0.2, 0.5, -0.4])
    # For spherical wrists, with TCP at the wrist center, J[:3, -3:] ≈ 0
    assert design.is_spherical_wrist_zero_block(robot, q, atol=1e-9)


# --------------------------------------------------------------------------- #
# Resolved-rates – least-squares consistency
# --------------------------------------------------------------------------- #

def test_resolved_rates_reproduces_task_velocity_in_ls_sense():
    robot = design.planar_2r(1.0, 1.0)
    q = np.array([0.4, -0.6])
    J = robot.jacobian_geometric(q)

    # Choose a target Xdot with translational components only for planar case
    xdot = np.array([0.1, -0.05, 0.0, 0.0, 0.0, 0.0])
    qdot = core.solvers.resolved_rates(J, xdot, damping=1e-6)

    # Check that J @ qdot ≈ xdot in least-squares sense
    residual = np.linalg.norm(J @ qdot - xdot)
    # Relative tolerance based on magnitude of xdot
    assert residual <= 1e-9 + 1e-7 * np.linalg.norm(xdot)


# --------------------------------------------------------------------------- #
# Newton–Raphson IK (position-only) – convergence for reachable point
# --------------------------------------------------------------------------- #

def test_newton_ik_position_only_converges():
    robot = design.planar_2r(1.0, 1.0)
    # Reachable point (not at singularity)
    target_p = np.array([1.4, 0.2, 0.0])
    q0 = np.array([0.1, 0.1])

    q_sol, info = core.solvers.newton_ik(
        robot,
        q0,
        x_target={"p": target_p},
        max_iter=100,
        tol=1e-10,
    )

    # Verify end-effector position close to target
    T = robot.fk(q_sol)["T_0e"]
    p = T[:3, 3]
    err = np.linalg.norm(p - target_p)
    assert info["converged"] is True
    assert err < 1e-6


# --------------------------------------------------------------------------- #
# Analytic Jacobian – basic properties
# --------------------------------------------------------------------------- #

def test_analytic_jacobian_shape_and_stability():
    robot = design.planar_2r(1.0, 1.0)
    q = np.array([0.2, -0.1])
    JA = robot.jacobian_analytic(q, euler="ZYX")
    Jg = robot.jacobian_geometric(q)

    assert JA.shape == Jg.shape
    # For the planar case (pure Rz rotations), the mapping is well-conditioned.
    # We at least ensure finite numbers and consistent dtype.
    assert np.all(np.isfinite(JA))
    assert JA.dtype == float
