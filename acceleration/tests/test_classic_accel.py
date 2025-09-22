# acceleration/tests/test_classic_accel.py
from __future__ import annotations

import math
import numpy as np
import pytest

# Under test: classic acceleration helpers
from acceleration.utils import (
    classic_accel,      # a = α×r + ω×(ω×r)
    S_from,             # linear operator S = [α] + [ω]^2 so a = S r
    omega_from_Rdot,    # body ω from Ṙ
    alpha_from_Rddot,   # body α from (R, Ṙ, R̈)
    homogeneous,        # convenience SE(3) constructor if needed elsewhere
)

# Spatial helper
from acceleration.tools.spatial import skew

# Euler helpers to generate smooth R(t)
from acceleration.tools.euler import (
    euler_matrix,           # R = euler_matrix(seq, angles)
)


# ----------------------------- utilities ------------------------------------

def _omega_from_Rdot_num(R, Rdot):
    """Reference body-ω from definition Rᵀ Ṙ = [ω]^."""
    S = R.T @ Rdot
    # vee(S) (skew to vector)
    return np.array([S[2,1] - S[1,2], S[0,2] - S[2,0], S[1,0] - S[0,1]]) * 0.5

def _alpha_from_Rddot_num(R, Rdot, Rddot):
    """Reference body-α from RᵀR̈ = [α]^ + [ω]^2."""
    w = _omega_from_Rdot_num(R, Rdot)
    w_hat = skew(w)
    S = R.T @ Rddot - (w_hat @ w_hat)
    return np.array([S[2,1] - S[1,2], S[0,2] - S[2,0], S[1,0] - S[0,1]]) * 0.5

def _euler_quadratic_trajectory(seq, q0, qd0, qdd0, dt):
    """
    Build R(-dt), R(0), R(+dt) using quadratic Euler trajectory:
      q(t) = q0 + qd0 t + 0.5 qdd0 t^2
    """
    qm = q0 - qd0*dt + 0.5*qdd0*(dt**2)
    qp = q0 + qd0*dt + 0.5*qdd0*(dt**2)
    Rm = euler_matrix(seq, qm)
    R0 = euler_matrix(seq, q0)
    Rp = euler_matrix(seq, qp)
    return Rm, R0, Rp


# ----------------------------- basic identities -----------------------------

def test_classic_accel_matches_double_cross():
    rng = np.random.default_rng(0)
    omega = rng.normal(size=3)
    alpha = rng.normal(size=3)
    r = rng.normal(size=3)

    # Direct double-cross: α×r + ω×(ω×r)
    a_ref = np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))

    # Function under test
    a_fun = classic_accel(alpha, omega, r)

    assert a_fun.shape == (3,)
    assert np.allclose(a_fun, a_ref, atol=1e-12)


def test_S_operator_matches_classic_and_blocks():
    rng = np.random.default_rng(1)
    omega = rng.normal(size=3)
    alpha = rng.normal(size=3)

    S = S_from(alpha, omega)   # S = [α] + [ω]^2
    # Check expected form:
    # [α] is skew-symmetric; [ω]^2 is symmetric negative semidefinite in general.
    A = skew(alpha)
    W2 = skew(omega) @ skew(omega)
    assert np.allclose(S, A + W2, atol=1e-12)

    # For random r, S r should equal classic_accel(α, ω, r)
    for _ in range(5):
        r = rng.normal(size=3)
        a1 = classic_accel(alpha, omega, r)
        a2 = S @ r
        assert np.allclose(a1, a2, atol=1e-12)


def test_zero_omega_alpha_zero_accel():
    r = np.array([0.3, -0.1, 0.2])
    a = classic_accel(np.zeros(3), np.zeros(3), r)
    assert np.allclose(a, np.zeros(3), atol=1e-15)


# --------------------------- numeric SO(3) check ----------------------------

@pytest.mark.parametrize("seq", ["ZYX", "XYZ", "ZXZ"])
def test_accel_from_euler_derivatives_in_inertial_frame(seq):
    """
    Let r be fixed in the rotating body frame. Then p(t) = R(t) r in the inertial frame obeys:
        p̈ = R ( α × r + ω × (ω × r) )
    where ω, α are **body** angular velocity/acceleration.
    """
    rng = np.random.default_rng(4)
    # Euler state at t=0 (moderate angles away from singularities for ZXZ)
    q0   = rng.uniform(-0.9, 0.9, size=3)
    qd0  = rng.normal(size=3) * 0.7
    qdd0 = rng.normal(size=3) * 0.4
    r_B  = rng.normal(size=3)

    dt = 1e-6

    # Build R(-dt), R(0), R(+dt) from a smooth Euler trajectory
    Rm, R0, Rp = _euler_quadratic_trajectory(seq, q0, qd0, qdd0, dt)

    # Numerical Ṙ, R̈ at t=0
    Rdot = (Rp - Rm) / (2*dt)
    Rddot = (Rp - 2*R0 + Rm) / (dt**2)

    # Compute body ω, α two ways: reference definitions and your helpers
    w_ref = _omega_from_Rdot_num(R0, Rdot)
    a_ref = _alpha_from_Rddot_num(R0, Rdot, Rddot)

    w_lib = omega_from_Rdot(R0, Rdot)
    a_lib = alpha_from_Rddot(R0, Rdot, Rddot)

    assert np.allclose(w_lib, w_ref, atol=1e-10)
    assert np.allclose(a_lib, a_ref, atol=1e-10)

    # Inertial-frame numeric p̈ = R̈ r
    pdd_num = Rddot @ r_B

    # Inertial-frame analytic p̈ = R (α×r + ω×(ω×r))
    a_body = classic_accel(a_lib, w_lib, r_B)
    pdd_analytic = R0 @ a_body

    assert np.allclose(pdd_num, pdd_analytic, atol=5e-4)


# --------------------------- frame-consistency ------------------------------

def test_body_vs_inertial_frame_projection():
    """
    If a_B = α×r + ω×(ω×r) (expressed in body frame), then the inertial vector is a_I = R a_B.
    Conversely, projecting back with Rᵀ should recover a_B.
    """
    rng = np.random.default_rng(8)
    # Random rotation
    ang = rng.uniform(-np.pi, np.pi, size=3)
    cz, sz = math.cos(ang[0]), math.sin(ang[0])
    cy, sy = math.cos(ang[1]), math.sin(ang[1])
    cx, sx = math.cos(ang[2]), math.sin(ang[2])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R = Rz @ Ry @ Rx

    omega_B = rng.normal(size=3)
    alpha_B = rng.normal(size=3)
    r_B = rng.normal(size=3)

    a_B = classic_accel(alpha_B, omega_B, r_B)
    a_I = R @ a_B
    # Project back
    a_B_back = R.T @ a_I
    assert np.allclose(a_B_back, a_B, atol=1e-12)
