# forward_kinematics/tests/test_core.py
from __future__ import annotations

import math
import numpy as np
import pytest

from forward_kinematics.core import Rotation, Transform, DHLink, MDHLink, PoELink, SerialChain
from forward_kinematics.utils import adjoint, almost_equal


def test_rotation_axis_angle_roundtrip():
    thx, thy, thz = 0.3, -0.4, 0.7
    R = Rotation.Rz(thz) @ Rotation.Ry(thy) @ Rotation.Rx(thx)
    phi, u = Rotation.axis_angle(R)
    # rebuild from axis–angle using Rodrigues
    ux, uy, uz = u
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]], float)
    R2 = np.eye(3) + np.sin(phi) * K + (1 - np.cos(phi)) * (K @ K)
    assert np.allclose(R, R2, atol=1e-9)


def test_transform_compose_and_inverse():
    R1 = Rotation.Rz(0.5); t1 = np.array([1.0, 0.2, -0.1])
    R2 = Rotation.Rx(-0.3); t2 = np.array([0.3, -0.4, 0.5])
    T1 = Transform(R1, t1)
    T2 = Transform(R2, t2)
    T = T1 @ T2
    I = T @ T.inv()
    assert np.allclose(I.as_matrix(), np.eye(4), atol=1e-9)

    # applying and then unapplying yields original point
    p = np.array([0.4, -0.2, 0.9])
    q = (T.apply(p))
    p_back = (T.inv().apply(q))
    assert np.allclose(p, p_back, atol=1e-9)


def test_dhlink_A_simple_case():
    # DH with a=1, alpha=0, d=0, theta=pi/2 should place x->0, y->1
    L = DHLink(a=1.0, alpha=0.0, d=0.0, joint_type="R")
    T = L.fk(math.pi/2).as_matrix()
    exp = np.array([[0.0, -1.0, 0.0, 0.0],
                    [1.0,  0.0, 0.0, 1.0],
                    [0.0,  0.0, 1.0, 0.0],
                    [0.0,  0.0, 0.0, 1.0]])
    assert np.allclose(T, exp, atol=1e-9)


def test_mdh_equals_dh_for_same_parameters():
    # Our closed form for MDH reduces to same numeric as DH for identical params here
    L_dh = DHLink(a=0.3, alpha=-0.5, d=0.2, theta_offset=0.1, joint_type="R")
    L_mdh = MDHLink(a=0.3, alpha=-0.5, d=0.2, theta_offset=0.1, joint_type="R")
    q = 0.35
    A_dh = L_dh.fk(q).as_matrix()
    A_mdh = L_mdh.fk(q).as_matrix()
    assert np.allclose(A_dh, A_mdh, atol=1e-12)


def test_serial_fk_planar_2r_rest_pose():
    # 2R: L1 at base (a=0), L2 carries l1, tool M carries l2 → at q=[0,0], x = l1 + l2
    l1, l2 = 1.0, 1.0
    L1 = DHLink(a=0.0, alpha=0.0, d=0.0, joint_type="R")
    L2 = DHLink(a=l1, alpha=0.0, d=0.0, joint_type="R")
    M = np.eye(4); M[0, 3] = l2
    robot = SerialChain([L1, L2], M=M)
    T = robot.fkine([0.0, 0.0]).as_matrix()
    exp = np.array([[1, 0, 0, l1 + l2],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], float)
    assert np.allclose(T, exp, atol=1e-9)


def test_poe_single_revolute_equals_Rz():
    w = np.array([0.0, 0.0, 1.0]); v = np.array([0.0, 0.0, 0.0])
    link = PoELink(w, v)
    robot = SerialChain([link], M=np.eye(4))
    th = 0.42
    T = robot.fkine([th]).as_matrix()
    c, s = math.cos(th), math.sin(th)
    exp = np.array([[c, -s, 0, 0],
                    [s,  c, 0, 0],
                    [0,  0, 1, 0],
                    [0,  0, 0, 1]], float)
    assert np.allclose(T, exp, atol=1e-12)


def test_jacobian_space_body_relation():
    # Use a small mixed chain (2R) and check J_b = Ad_T^{-1} J_s
    L1 = DHLink(a=0.0, alpha=0.0, d=0.0, joint_type="R")
    L2 = DHLink(a=0.7, alpha=0.0, d=0.0, joint_type="R")
    M = np.eye(4); M[0, 3] = 0.5
    robot = SerialChain([L1, L2], M=M)

    q = np.array([0.2, -0.4])
    Js = robot.jacobian_space(q)
    Jb = robot.jacobian_body(q)
    T = robot.fkine(q).as_matrix()
    Ad_inv = np.linalg.inv(adjoint(T))
    assert np.allclose(Jb, Ad_inv @ Js, atol=1e-9)
    # dimensions
    assert Js.shape == (6, 2)
    assert Jb.shape == (6, 2)


def test_fkine_input_length_validation():
    L1 = DHLink(a=0.0, alpha=0.0, d=0.0)
    robot = SerialChain([L1])
    with pytest.raises(ValueError):
        _ = robot.fkine([0.1, 0.2])  # wrong length
