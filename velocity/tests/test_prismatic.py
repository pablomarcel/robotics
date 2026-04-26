# velocity/tests/test_prismatic.py
"""
Prismatic joint behavior tests.

Covers:
- FK parameterization for a prismatic joint: d = d_offset + q for type 'P'
- Jacobian column formula for prismatic: [k_i ; 0]
- Mixed R–P chain Jacobian columns (first revolute, then prismatic)
"""

from __future__ import annotations

import numpy as np

from velocity.core import JointDH, DHRobot
from velocity import design


def test_single_prismatic_fk_and_jacobian():
    # One prismatic joint along +z at base, with an x-offset tool to make position observable.
    j1 = JointDH(name="p1", joint_type="P", alpha=0.0, a=0.0, d=0.2, theta=0.0)  # d = 0.2 + q
    tool = np.eye(4)
    tool[0, 3] = 0.3  # shift TCP along x so p changes only in z via prismatic d
    robot = DHRobot([j1], tool=tool, name="P1")

    # Configuration: q increases d
    q = np.array([0.5])  # d_total = 0.2 + 0.5 = 0.7
    fk = robot.fk(q)
    T = fk["T_0e"]
    # End-effector position should be (0.3, 0, 0.7)
    p = T[:3, 3]
    assert np.allclose(p, np.array([0.3, 0.0, 0.7]), atol=1e-12)

    # Jacobian: for a prismatic joint, column = [k ; 0], k is base z-axis here
    J = robot.jacobian_geometric(q)
    assert J.shape == (6, 1)
    k = np.array([0.0, 0.0, 1.0])
    expected_col = np.r_[k, [0.0, 0.0, 0.0]]
    assert np.allclose(J[:, 0], expected_col, atol=1e-12)


def test_revolute_then_prismatic_columns():
    # 2-DOF chain: R about z at base, then P along z of joint 2 (standard DH).
    # Link 1 with a=0.4 so p_i locations are distinct.
    j1 = JointDH(name="r1", joint_type="R", alpha=0.0, a=0.4, d=0.0, theta=0.0)      # revolute
    j2 = JointDH(name="p2", joint_type="P", alpha=0.0, a=0.0, d=0.1, theta=0.0)      # prismatic
    robot = DHRobot([j1, j2], tool=np.eye(4), name="R_then_P")

    q = np.array([0.3, 0.2])  # theta1=0.3, d2=0.1+0.2
    J = robot.jacobian_geometric(q)

    # Column 1 (revolute): [k1 × (pe - p1) ; k1]
    # For standard DH, k1 is base z rotated by Rz(theta1) -> still base z (since rotation_kinematics about z).
    # But p1 is at end of joint 1 frame after T1; compute expected numerically via small motion check
    # to stay robust to any modeling nuances.
    eps = 1e-8
    J_num = np.zeros_like(J)
    # revolute numerical column via finite-difference on q1
    q_eps = q.copy()
    q_eps[0] += eps
    p_plus = robot.fk(q_eps)["T_0e"][:3, 3]
    q_eps[0] -= 2 * eps
    p_minus = robot.fk(q_eps)["T_0e"][:3, 3]
    v_fd = (p_plus - p_minus) / (2 * eps)
    # angular velocity about z for small delta q1 is k1 = [0,0,1]
    w1 = np.array([0.0, 0.0, 1.0])
    J_num[:3, 0] = v_fd
    J_num[3:, 0] = w1

    # prismatic column should be [k2 ; 0], where k2 is joint-2 z expressed in base.
    # For this DH with alpha2=0, k2 is also base z.
    k2 = np.array([0.0, 0.0, 1.0])
    J_num[:, 1] = np.r_[k2, [0.0, 0.0, 0.0]]

    assert np.allclose(J, J_num, atol=1e-6)


def test_planar_rp_model_via_design_and_prismatic_column():
    # Use design.planar_2r as base and swap second to prismatic to ensure utility compatibility.
    base = design.planar_2r(0.8, 0.6)
    # mutate: convert j2 to prismatic keeping same a for geometric visibility
    j1 = base.joints[0]
    j2p = JointDH(name="j2p", joint_type="P", alpha=0.0, a=0.6, d=0.0, theta=0.0)
    robot = DHRobot([j1, j2p], tool=np.eye(4), name="R_P_planar")

    q = np.array([0.25, 0.1])
    J = robot.jacobian_geometric(q)

    # Prismatic column (second) should have zero angular part
    assert np.allclose(J[3:, 1], 0.0, atol=1e-12)
    # Translational part should be unit z rotated into base; with alpha=0 and standard frames it's base z.
    assert np.allclose(J[:3, 1], np.array([0.0, 0.0, 1.0]), atol=1e-12)
