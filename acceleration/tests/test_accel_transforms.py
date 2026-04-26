from __future__ import annotations
import numpy as np
from acceleration.tools.spatial import skew
from acceleration.utils import (
    classic_accel, S_from, omega_from_Rdot, alpha_from_Rddot
)

def _vee(S):
    return np.array([S[2,1]-S[1,2], S[0,2]-S[2,0], S[1,0]-S[0,1]]) * 0.5

def test_skew_vee_roundtrip():
    rng = np.random.default_rng(0)
    v = rng.normal(size=3)
    S = skew(v)
    v2 = _vee(S)
    assert np.allclose(v, v2, atol=1e-12)

def test_S_operator_matches_classic():
    rng = np.random.default_rng(1)
    alpha = rng.normal(size=3)
    omega = rng.normal(size=3)
    S = S_from(alpha, omega)
    A = skew(alpha)
    W2 = skew(omega) @ skew(omega)
    assert np.allclose(S, A + W2, atol=1e-12)

    # for random r, S r equals classic_accel
    for _ in range(5):
        r = rng.normal(size=3)
        assert np.allclose(S @ r, classic_accel(alpha, omega, r), atol=1e-12)

def test_omega_alpha_from_R_derivatives_consistency():
    # small rotation_kinematics trajectory around random axis
    rng = np.random.default_rng(2)
    u = rng.normal(size=3); u = u / np.linalg.norm(u)
    w = 0.7 * u
    a = -0.3 * u
    dt = 1e-6

    # R(t) = exp([w] t + 0.5 [a] t^2) ≈ I + [w] t + 0.5([a]+[w]^2) t^2
    W = skew(w); A = skew(a)
    Rm = np.eye(3) - W*dt + 0.5*(A + W@W)*(dt**2)
    R0 = np.eye(3)
    Rp = np.eye(3) + W*dt + 0.5*(A + W@W)*(dt**2)

    Rdot = (Rp - Rm)/(2*dt)
    Rdd  = (Rp - 2*R0 + Rm)/(dt**2)

    w_lib = omega_from_Rdot(R0, Rdot)
    a_lib = alpha_from_Rddot(R0, Rdot, Rdd)
    assert np.allclose(w_lib, w, atol=5e-9)
    assert np.allclose(a_lib, a, atol=5e-8)
