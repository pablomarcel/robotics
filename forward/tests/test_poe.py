# forward/tests/test_poe.py
from __future__ import annotations

import numpy as np

from forward.core import PoELink, SerialChain
from forward.utils import adjoint


def test_single_prismatic_translation():
    # Prismatic along +x in space frame: omega=0, v = x̂
    link = PoELink(omega=np.zeros(3), v=np.array([1.0, 0.0, 0.0]))
    robot = SerialChain([link], M=np.eye(4))
    q = [0.25]
    T = robot.fkine(q).as_matrix()
    exp = np.array([[1, 0, 0, 0.25],
                    [0, 1, 0, 0.0 ],
                    [0, 0, 1, 0.0 ],
                    [0, 0, 0, 1.0 ]], float)
    assert np.allclose(T, exp, atol=1e-12)


def test_two_revolute_z_then_x_rotation_composition():
    # Two revolutes with screws fixed in SPACE:
    # S1: about z at origin → Rz(q1)
    # S2: about x at origin → Rx(q2)
    S1 = PoELink(omega=np.array([0.0, 0.0, 1.0]), v=np.zeros(3))
    S2 = PoELink(omega=np.array([1.0, 0.0, 0.0]), v=np.zeros(3))
    robot = SerialChain([S1, S2], M=np.eye(4))

    q1, q2 = 0.3, -0.4
    T = robot.fkine([q1, q2]).as_matrix()

    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    Rz = np.array([[c1, -s1, 0],
                   [s1,  c1, 0],
                   [0 ,   0, 1]], float)
    Rx = np.array([[1, 0,  0],
                   [0, c2, -s2],
                   [0, s2,  c2]], float)
    exp = np.eye(4)
    exp[:3, :3] = Rz @ Rx
    assert np.allclose(T, exp, atol=1e-12)


def test_space_jacobian_matches_adjoint_accumulation():
    # Chain: S1 (z@origin), S2 (x@origin), S3 (prismatic along y)
    S1 = PoELink(omega=np.array([0.0, 0.0, 1.0]), v=np.zeros(3))
    S2 = PoELink(omega=np.array([1.0, 0.0, 0.0]), v=np.zeros(3))
    S3 = PoELink(omega=np.zeros(3), v=np.array([0.0, 1.0, 0.0]))
    robot = SerialChain([S1, S2, S3], M=np.eye(4))

    q = np.array([0.6, -0.2, 0.05])
    J = robot.jacobian_space(q)

    # Manual adjoint accumulation per PoE theory:
    # J[:,1] = S1
    # J[:,2] = Ad(exp(S1*q1)) S2
    # J[:,3] = Ad(exp(S1*q1) exp(S2*q2)) S3
    S1v = np.hstack([S1.omega, S1.v]).reshape(6, 1)
    S2v = np.hstack([S2.omega, S2.v]).reshape(6, 1)
    S3v = np.hstack([S3.omega, S3.v]).reshape(6, 1)

    T1 = S1.fk(q[0]).as_matrix()
    T2 = S2.fk(q[1]).as_matrix()

    J1 = S1v
    J2 = adjoint(T1) @ S2v
    J3 = adjoint(T1 @ T2) @ S3v

    J_exp = np.hstack([J1, J2, J3])
    assert np.allclose(J, J_exp, atol=1e-12)


def test_home_pose_M_applied_at_zero_configuration():
    # With q=0, FK should equal M exactly.
    S = PoELink(omega=np.array([0.0, 0.0, 1.0]), v=np.zeros(3))
    M = np.eye(4); M[:3, 3] = np.array([0.1, 0.2, 0.3])
    robot = SerialChain([S], M=M)
    T = robot.fkine([0.0]).as_matrix()
    assert np.allclose(T, M, atol=1e-12)
