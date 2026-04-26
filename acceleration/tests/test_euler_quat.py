# acceleration/tests/test_euler_quat.py
from __future__ import annotations

import math
import numpy as np
import pytest

from acceleration.tools.euler import (
    euler_matrix,            # R = euler_matrix(seq, angles)
    euler_rates_matrix,      # E = euler_rates_matrix(seq, angles) -> ω = E q̇
    euler_accel_matrix,      # Ė = euler_accel_matrix(seq, angles, rates) so α = E q̈ + Ė q̇
)
from acceleration.tools.quaternion import (
    quat_normalize,
    quat_to_R,
    R_to_quat,
    quat_multiply,
    quat_kinematics_matrix,  # Q(q) s.t. q̇ = 0.5 * Q(q) ω
    omega_from_quat_rates,   # ω(q, q̇)
    alpha_from_quat_rates,   # α(q, q̇, q̈)
)

# ---------------------------------------------------------------------------
# Helpers (SO(3) derivatives → ω, α)
# ---------------------------------------------------------------------------

def _vee(S):
    # vee of a 3x3 skew
    return np.array([S[2,1] - S[1,2], S[0,2] - S[2,0], S[1,0] - S[0,1]]) * 0.5

def _omega_from_Rdot(R, Rdot):
    # R^T Rdot = [ω]^
    return _vee(R.T @ Rdot)

def _alpha_from_Rddot(R, Rdot, Rddot):
    # R^T Rddot = [α]^ + [ω]^2  →  [α]^ = R^T Rddot - [ω]^2
    omega = _omega_from_Rdot(R, Rdot)
    o_hat = np.array([[0, -omega[2], omega[1]],
                      [omega[2], 0, -omega[0]],
                      [-omega[1], omega[0], 0]], dtype=float)
    S = R.T @ Rddot - o_hat @ o_hat
    return _vee(S)

def _skew(v):
    x, y, z = np.asarray(v, float).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)

# Numerically differentiate R(·) given q(t), q̇(t), q̈(t) via finite differences
def _num_R_derivatives(seq, q, qd, qdd, dt=1e-6):
    """
    q, qd, qdd are 3-vectors (Euler generalized coords, rates, accels).
    Centered differences for Ṙ and R̈ at t=0 with a quadratic trajectory:
        q(t) ≈ q + qd t + 0.5 qdd t^2
    """
    qm = q - qd*dt + 0.5*qdd*(dt**2)
    qp = q + qd*dt + 0.5*qdd*(dt**2)
    R0 = euler_matrix(seq, q)
    Rm = euler_matrix(seq, qm)
    Rp = euler_matrix(seq, qp)
    Rdot = (Rp - Rm) / (2*dt)
    Rddot = (Rp - 2*R0 + Rm) / (dt**2)
    return R0, Rdot, Rddot

# ---------------------------------------------------------------------------
# Euler: rotation_kinematics orthogonality & det
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq", ["ZYX", "XYZ", "ZXZ", "ZYZ"])
def test_euler_matrix_is_rotation(seq):
    rng = np.random.default_rng(0)
    q = rng.uniform(low=-np.pi, high=np.pi, size=3)
    R = euler_matrix(seq, q)
    assert R.shape == (3, 3)
    # Orthogonality & det +1
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    det = np.linalg.det(R)
    assert np.allclose(det, 1.0, atol=1e-12)

# ---------------------------------------------------------------------------
# Euler rates: ω = E(q) q̇ against numerical ω from Ṙ
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq", ["ZYX", "XYZ", "ZXZ", "ZYZ"])
def test_euler_rates_match_numerical_omega(seq):
    rng = np.random.default_rng(4)
    q = rng.uniform(-1.0, 1.0, size=3)
    qd = rng.normal(size=3)

    E = euler_rates_matrix(seq, q)
    omega_analytic = E @ qd

    # numerical ω from Ṙ
    R, Rdot, _ = _num_R_derivatives(seq, q, qd, np.zeros(3))
    omega_num = _omega_from_Rdot(R, Rdot)
    assert np.allclose(omega_analytic, omega_num, atol=1e-6)

# ---------------------------------------------------------------------------
# Euler accelerations: α = E q̈ + Ė q̇ vs numerical α from R̈
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq", ["ZYX", "XYZ", "ZXZ", "ZYZ"])
def test_euler_accel_matches_numerical_alpha(seq):
    rng = np.random.default_rng(6)
    q   = rng.uniform(-1.0, 1.0, size=3)
    qd  = rng.normal(size=3)
    qdd = rng.normal(size=3)

    E  = euler_rates_matrix(seq, q)
    Ed = euler_accel_matrix(seq, q, qd)  # Ė(q, q̇)
    alpha_analytic = E @ qdd + Ed @ qd

    R, Rdot, Rddot = _num_R_derivatives(seq, q, qd, qdd)
    alpha_num = _alpha_from_Rddot(R, Rdot, Rddot)
    assert np.allclose(alpha_analytic, alpha_num, atol=5e-5)

# ---------------------------------------------------------------------------
# Quaternion <-> Rotation round-trip (up to sign)
# ---------------------------------------------------------------------------

def _quat_sign_free_err(q_expected, q_got):
    q_expected = quat_normalize(np.asarray(q_expected, float).reshape(4))
    q_got = quat_normalize(np.asarray(q_got, float).reshape(4))
    return min(np.linalg.norm(q_expected - q_got), np.linalg.norm(q_expected + q_got))

def test_quaternion_roundtrip():
    rng = np.random.default_rng(10)
    for _ in range(50):
        # random unit quaternion
        v = rng.normal(size=4)
        q = v / np.linalg.norm(v)
        R = quat_to_R(q)
        q2 = R_to_quat(R)
        assert _quat_sign_free_err(q, q2) < 1e-10

def test_quaternion_composition_matches_rotation_product():
    rng = np.random.default_rng(12)
    # two random unit quaternions
    def rand_q():
        v = rng.normal(size=4)
        return v / np.linalg.norm(v)
    q1 = rand_q()
    q2 = rand_q()
    R1 = quat_to_R(q1)
    R2 = quat_to_R(q2)
    R12 = R1 @ R2
    q12 = quat_multiply(q1, q2)
    R12_from_q = quat_to_R(q12)
    assert np.allclose(R12, R12_from_q, atol=1e-12)

# ---------------------------------------------------------------------------
# Quaternion kinematics: ω, α from (q, q̇, q̈) vs numerical Ṙ, R̈
# ---------------------------------------------------------------------------

def _finite_diff_quat(q_minus, q_plus, dt):
    """
    Central difference q̇ ≈ (q+ - q-) / (2dt), adjusted for sign continuity
    to avoid jumps at antipodes (since q and -q represent same rotation_kinematics).
    """
    qm = quat_normalize(q_minus)
    qp = quat_normalize(q_plus)
    # enforce sign continuity: pick sign of qp to be close to qm
    if np.dot(qm, qp) < 0.0:
        qp = -qp
    qdot = (qp - qm) / (2 * dt)
    return qdot

def _second_diff_quat(q_minus, q0, q_plus, dt):
    qm = quat_normalize(q_minus)
    q0 = quat_normalize(q0)
    qp = quat_normalize(q_plus)
    # enforce continuity between neighbors
    if np.dot(qm, q0) < 0.0: q0 = -q0
    if np.dot(q0, qp) < 0.0: qp = -qp
    qdd = (qp - 2*q0 + qm) / (dt**2)
    return qdd

@pytest.mark.parametrize("seq", ["ZYX", "XYZ"])
def test_quaternion_omega_alpha_match_numerical(seq):
    rng = np.random.default_rng(16)
    q_e = rng.uniform(-0.8, 0.8, size=3)           # euler angles at t=0
    qd_e = rng.normal(size=3) * 0.5                # rates
    qdd_e = rng.normal(size=3) * 0.3               # accels

    dt = 1e-6

    # Build R(t) via Euler, then get q(t) from R(t)
    qm_e = q_e - qd_e*dt + 0.5*qdd_e*(dt**2)
    qp_e = q_e + qd_e*dt + 0.5*qdd_e*(dt**2)

    Rm = euler_matrix(seq, qm_e)
    R0 = euler_matrix(seq, q_e)
    Rp = euler_matrix(seq, qp_e)

    # numerical ω, α from Ṙ, R̈
    Rdot = (Rp - Rm) / (2*dt)
    Rddot = (Rp - 2*R0 + Rm) / (dt**2)
    omega_num = _omega_from_Rdot(R0, Rdot)
    alpha_num = _alpha_from_Rddot(R0, Rdot, Rddot)

    # Convert to quaternions at t=-dt, 0, +dt
    q_minus = R_to_quat(Rm)
    q0 = R_to_quat(R0)
    q_plus = R_to_quat(Rp)

    # Central differences for q̇, q̈
    qdot = _finite_diff_quat(q_minus, q_plus, dt)
    qddot = _second_diff_quat(q_minus, q0, q_plus, dt)

    # Recovered ω, α from quaternion kinematics
    omega_q = omega_from_quat_rates(q0, qdot)
    alpha_q = alpha_from_quat_rates(q0, qdot, qddot)

    assert np.allclose(omega_q, omega_num, atol=5e-6)
    assert np.allclose(alpha_q, alpha_num, atol=5e-4)

def test_quat_kinematics_matrix_consistency():
    """
    Check q̇ ≈ 0.5 Q(q) ω and inverse ω(q, q̇) round-trip for small motions.
    """
    rng = np.random.default_rng(20)
    # random rotation_kinematics → quaternion
    ang = rng.uniform(-np.pi, np.pi, size=3)
    # build R from ZYX and convert to q
    cz, sz = math.cos(ang[0]), math.sin(ang[0])
    cy, sy = math.cos(ang[1]), math.sin(ang[1])
    cx, sx = math.cos(ang[2]), math.sin(ang[2])
    Rz = np.array([[cz, -sz, 0],[sz, cz, 0],[0, 0, 1]])
    Ry = np.array([[cy, 0, sy],[0, 1, 0],[-sy, 0, cy]])
    Rx = np.array([[1, 0, 0],[0, cx, -sx],[0, sx, cx]])
    R = Rz @ Ry @ Rx
    q = R_to_quat(R)

    omega = rng.normal(size=3) * 0.1
    Q = quat_kinematics_matrix(q)
    qdot = 0.5 * (Q @ omega)

    # invert back to ω using helper
    omega_back = omega_from_quat_rates(q, qdot)
    assert np.allclose(omega_back, omega, atol=1e-10)
