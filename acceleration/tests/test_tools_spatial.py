# acceleration/tests/test_tools_spatial.py
from __future__ import annotations

import math
import numpy as np
import pytest

# Under test
from acceleration.tools.spatial import (
    skew, vex,
    adjoint,
    motion_xform,     # X(T)
    force_xform,      # X*(T) = (X(T)^{-1})^T
    cross_motion,     # v×  (6×6)  acting on motion_kinematics vectors
    cross_force,      # v×* (6×6)  acting on force vectors
    spatial_inertia,  # I = [[Ic + m[c]^[c]^T, m[c]^], [-m[c]^, mI3]]
)

# Utilities from the project (used for constructing SE(3) transforms)
from acceleration.utils import homogeneous


def rand_so3(rng: np.random.Generator) -> np.ndarray:
    """
    Random rotation_kinematics via axis-angle.
    """
    u = rng.normal(size=3)
    theta = np.linalg.norm(u)
    if theta < 1e-12:
        return np.eye(3)
    k = u / theta
    K = skew(k)
    c, s = math.cos(theta), math.sin(theta)
    return np.eye(3) * c + (1 - c) * np.outer(k, k) + s * K


def rand_se3(rng: np.random.Generator) -> np.ndarray:
    R = rand_so3(rng)
    p = rng.uniform(-1.0, 1.0, size=3)
    return homogeneous(R, p)


# ---------------------------------------------------------------------------
# Basic hat/vee
# ---------------------------------------------------------------------------

def test_skew_vee_inverse():
    rng = np.random.default_rng(42)
    v = rng.normal(size=3)
    S = skew(v)
    assert S.shape == (3, 3)
    # Skew is antisymmetric
    assert np.allclose(S + S.T, np.zeros((3, 3)), atol=1e-12)
    # vex(skew(v)) == v  (up to tiny numerical noise)
    assert np.allclose(vex(S), v, atol=1e-12)


# ---------------------------------------------------------------------------
# Adjoint (SE(3) → Ad_T ∈ R^{6×6})
# ---------------------------------------------------------------------------

def test_adjoint_identity_and_composition():
    I4 = np.eye(4)
    AdI = adjoint(I4)
    assert AdI.shape == (6, 6)
    assert np.allclose(AdI, np.eye(6), atol=1e-12)

    rng = np.random.default_rng(0)
    T1 = rand_se3(rng)
    T2 = rand_se3(rng)
    T12 = T1 @ T2
    Ad1 = adjoint(T1)
    Ad2 = adjoint(T2)
    Ad12 = adjoint(T12)
    # Homomorphism: Ad_{T1 T2} = Ad_{T1} Ad_{T2}
    assert np.allclose(Ad12, Ad1 @ Ad2, atol=1e-12)


def test_adjoint_block_structure():
    rng = np.random.default_rng(1)
    T = rand_se3(rng)
    R = T[:3, :3]
    p = T[:3, 3]
    p_hat = skew(p)
    Ad = adjoint(T)

    # Expected block structure:
    # Ad_T = [[R, 0],
    #         [p^ R, R]]
    top_left = Ad[:3, :3]
    top_right = Ad[:3, 3:]
    bot_left = Ad[3:, :3]
    bot_right = Ad[3:, 3:]

    assert np.allclose(top_left, R, atol=1e-12)
    assert np.allclose(top_right, np.zeros((3, 3)), atol=1e-12)
    assert np.allclose(bot_left, p_hat @ R, atol=1e-12)
    assert np.allclose(bot_right, R, atol=1e-12)


# ---------------------------------------------------------------------------
# Motion/Force transforms duality & power invariance
# ---------------------------------------------------------------------------

def test_motion_force_duality_and_power_invariance():
    rng = np.random.default_rng(7)
    T = rand_se3(rng)
    X = motion_xform(T)
    Xstar = force_xform(T)

    # Duality: X* = (X^{-1})^T
    X_inv_T = np.linalg.inv(X).T
    assert np.allclose(Xstar, X_inv_T, atol=1e-12)

    # Power invariance: v^T f == (X v)^T (X* f)
    v = rng.normal(size=6)
    f = rng.normal(size=6)
    lhs = float(v.T @ f)
    rhs = float((X @ v).T @ (Xstar @ f))
    assert np.allclose(lhs, rhs, atol=1e-12)


# ---------------------------------------------------------------------------
# Cross operators
# ---------------------------------------------------------------------------

def test_cross_duality_and_action():
    rng = np.random.default_rng(13)
    v = rng.normal(size=6)  # spatial motion_kinematics
    w = rng.normal(size=6)

    vx = cross_motion(v)      # 6x6
    vx_star = cross_force(v)  # 6x6

    # Duality: v×* = -(v×)^T
    assert np.allclose(vx_star, -(vx.T), atol=1e-12)

    # Action properties (linearity)
    a = 0.37
    b = -1.2
    v2 = a * v + b * w
    vx2 = cross_motion(v2)
    assert np.allclose(vx2, a * cross_motion(v) + b * cross_motion(w), atol=1e-12)


# ---------------------------------------------------------------------------
# Spatial inertia
# ---------------------------------------------------------------------------

def test_spatial_inertia_block_form_and_symmetry():
    rng = np.random.default_rng(21)
    m = float(abs(rng.normal()) + 0.1)  # ensure positive
    c = rng.normal(size=3)              # CoM expressed in the body frame
    Ic = np.eye(3) * 0.2                # simple SPD rotational inertia about CoM

    I = spatial_inertia(m=m, com=c, Ic=Ic)
    assert I.shape == (6, 6)
    assert np.allclose(I, I.T, atol=1e-12)  # symmetric

    c_hat = skew(c)
    # Expected block structure:
    # I = [[Ic + m c^ c^T,   m c^],
    #      [ -m c^,          m I3]]
    TL = I[:3, :3]
    TR = I[:3, 3:]
    BL = I[3:, :3]
    BR = I[3:, 3:]

    assert np.allclose(TR, m * c_hat, atol=1e-12)
    assert np.allclose(BL, -m * c_hat, atol=1e-12)
    assert np.allclose(BR, m * np.eye(3), atol=1e-12)
    assert np.allclose(TL, Ic + m * (c_hat @ c_hat.T), atol=1e-12)


def test_spatial_inertia_positive_semidefinite():
    rng = np.random.default_rng(23)
    m = 2.5
    c = np.array([0.1, -0.2, 0.05])
    Ic = np.diag([0.03, 0.04, 0.05])   # SPD about CoM

    I = spatial_inertia(m=m, com=c, Ic=Ic)
    # For any spatial motion_kinematics v, kinetic energy 0.5 vᵀ I v >= 0
    for _ in range(100):
        v = rng.normal(size=6)
        ke = float(v.T @ (I @ v))
        assert ke >= -1e-10  # numerical slack only


def test_spatial_inertia_transform_consistency():
    """
    Spatial inertia must transform with the **force** transform when twists map by X.

    Conventions used in these tests:
      - Twists (spatial motion_kinematics):   v_B = X_AB v_A
      - Wrenches (spatial force):  f_B = X*_AB f_A  with  X*_AB = (X_AB^{-1})^T

    Power/kinetic energy invariance for all v_A implies:
        v_A^T I_A v_A = v_B^T I_B v_B = v_A^T (X_AB^T I_B X_AB) v_A
      → X_AB^T I_B X_AB = I_A
      → I_B = X_AB^{-T} I_A X_AB^{-1} = X*_AB I_A X_BA

    where X_BA = X_AB^{-1} is the motion_kinematics transform for T_BA = T_AB^{-1}.
    """
    rng = np.random.default_rng(29)
    m = 1.7
    c_A = np.array([0.2, 0.0, -0.1])
    Ic_A = np.diag([0.02, 0.03, 0.04])

    I_A = spatial_inertia(m=m, com=c_A, Ic=Ic_A)

    # Random transform from A to B
    T_A_to_B = rand_se3(rng)
    X_AB = motion_xform(T_A_to_B)
    Xstar_AB = force_xform(T_A_to_B)

    # Build X_BA without directly inverting the 6x6: use the inverse SE(3)
    T_B_to_A = np.linalg.inv(T_A_to_B)
    X_BA = motion_xform(T_B_to_A)

    # Correct inertia mapping: I_B = X*_AB I_A X_BA
    I_B = Xstar_AB @ I_A @ X_BA

    # Power/kinetic-energy invariance check with random twists
    for _ in range(20):
        v_A = rng.normal(size=6)
        v_B = X_AB @ v_A
        pwr_A = float(v_A.T @ (I_A @ v_A))
        pwr_B = float(v_B.T @ (I_B @ v_B))
        assert np.allclose(pwr_A, pwr_B, atol=1e-10)


# ---------------------------------------------------------------------------
# Motion/force transform shapes & edge cases
# ---------------------------------------------------------------------------

def test_motion_force_shapes_and_identities():
    T = np.eye(4)
    X = motion_xform(T)
    Xstar = force_xform(T)
    assert X.shape == (6, 6)
    assert Xstar.shape == (6, 6)
    assert np.allclose(X, np.eye(6), atol=1e-12)
    assert np.allclose(Xstar, np.eye(6), atol=1e-12)

    # Pure translation should affect only lower-left block of X
    p = np.array([0.3, -0.1, 0.25])
    R = np.eye(3)
    T = homogeneous(R, p)
    X = motion_xform(T)
    p_hat = skew(p)
    # X = [[R, 0], [p^ R, R]] with R = I
    assert np.allclose(X[:3, :3], np.eye(3), atol=1e-12)
    assert np.allclose(X[:3, 3:], np.zeros((3, 3)), atol=1e-12)
    assert np.allclose(X[3:, :3], p_hat, atol=1e-12)
    assert np.allclose(X[3:, 3:], np.eye(3), atol=1e-12)
