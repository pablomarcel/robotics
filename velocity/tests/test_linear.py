# velocity/tests/test_linear.py
"""
Linear algebra & math utility tests for the Velocity Kinematics module.

Covers:
- skew() reproduces cross products via matrix multiply
- SE(3)/SO(3) helpers: rotx/roty/rotz orthogonality, r2t/t2r roundtrip, mmul
- pinv_damped() is stable at singularity and reproduces task rates in LS sense
"""

from __future__ import annotations

import numpy as np
import pytest

from velocity import design, utils


# --------------------------------------------------------------------------- #
# skew() vs cross product
# --------------------------------------------------------------------------- #

def test_skew_matches_cross_product():
    a = np.array([0.3, -0.7, 1.2])
    b = np.array([1.1, 0.5, -0.4])
    S = utils.skew(a)
    lhs = S @ b
    rhs = np.cross(a, b)
    assert np.allclose(lhs, rhs, atol=1e-12)


# --------------------------------------------------------------------------- #
# SO(3)/SE(3) helpers
# --------------------------------------------------------------------------- #

def test_rotation_matrices_are_orthonormal():
    for ang in [0.0, 0.2, -1.1, 2.7]:
        for R in (utils.rotx(ang), utils.roty(ang), utils.rotz(ang)):
            RtR = R.T @ R
            det = np.linalg.det(R)
            assert np.allclose(RtR, np.eye(3), atol=1e-12)
            assert np.allclose(det, 1.0, atol=1e-12)

def test_r2t_t2r_roundtrip_and_transl():
    R = utils.rotz(0.3) @ utils.rotx(-0.2)
    p = np.array([0.4, -0.1, 0.7])
    T = utils.r2t(R, p)
    R2, p2 = utils.t2r(T)
    assert np.allclose(R2, R, atol=1e-12)
    assert np.allclose(p2, p, atol=1e-12)

    Ttx = utils.transl(0.4, -0.1, 0.7)
    _, p3 = utils.t2r(Ttx)
    assert np.allclose(p3, p, atol=1e-12)

def test_mmul_left_to_right_multiplication():
    A = utils.rotz(0.2)
    B = utils.rotx(-0.1)
    C = utils.roty(0.3)
    left_to_right = utils.mmul(A, B, C)
    vanilla = A @ B @ C
    assert np.allclose(left_to_right, vanilla, atol=1e-12)


# --------------------------------------------------------------------------- #
# pinv_damped stability near singularities
# --------------------------------------------------------------------------- #

def test_pinv_damped_is_stable_at_singular_configuration():
    # Planar 2R straight configuration is singular for translation in certain directions.
    robot = design.planar_2r(1.0, 1.0)
    q_sing = np.array([0.0, 0.0])  # arm straight along +x
    J = robot.jacobian_geometric(q_sing)

    # Task velocity primarily along x is feasible; along y at full extension is harder.
    xdot = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])

    # Damped solution must be finite and produce small residual in LS sense.
    lam = 1e-4
    Jplus = utils.pinv_damped(J, lam=lam)
    qdot = Jplus @ xdot

    assert np.all(np.isfinite(qdot))
    resid = np.linalg.norm(J @ qdot - xdot)
    # Residual should be bounded roughly by O(lam) scaling
    assert resid <= 1e-4 + 1e-2 * np.linalg.norm(xdot)


def test_pinv_damped_converges_to_lstsq_when_well_conditioned():
    # Pick a well-conditioned configuration away from singularities
    robot = design.planar_2r(0.8, 0.6)
    q = np.array([0.3, -0.5])
    J = robot.jacobian_geometric(q)
    xdot = np.array([0.1, -0.02, 0.0, 0.0, 0.0, 0.0])

    # Damping extremely small ≈ pseudo-inverse_kinematics
    q_damped = utils.pinv_damped(J, lam=1e-9) @ xdot

    # Compare with least-squares solution via SVD-based pinv
    q_pinv = np.linalg.pinv(J) @ xdot

    assert np.allclose(q_damped, q_pinv, atol=1e-9)


# --------------------------------------------------------------------------- #
# normalize() edge cases
# --------------------------------------------------------------------------- #

def test_normalize_behaviour_on_zero_and_nonzero():
    v_zero = np.zeros(3)
    v_unit = utils.normalize(v_zero)
    assert np.allclose(v_unit, v_zero)

    v = np.array([3.0, 4.0, 0.0])
    vn = utils.normalize(v)
    assert np.allclose(np.linalg.norm(vn), 1.0, atol=1e-12)
    # direction preserved (parallel)
    cross = np.linalg.norm(np.cross(v, vn))
    assert cross < 1e-12
