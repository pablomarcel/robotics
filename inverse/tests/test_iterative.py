# inverse/tests/test_iterative.py
# Pytest suite focused on the iterative IK solver and utilities.

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from inverse.core import SerialChain, DHLink, IterativeIK, Transform
from inverse import design as design_mod
from inverse import utils as U


def _planar_2r_chain(l1=1.0, l2=1.0) -> SerialChain:
    # Use the design helper to ensure consistency with the package.
    return design_mod.planar_2r(l1, l2)


def _T_xy(x: float, y: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    return T


# ------------------------------ utilities ---------------------------------

def test_pose_error_small_vs_so3_small_angles():
    """Orientation error should agree between 'small' and 'so3' for small angles."""
    # Tiny rotation_kinematics about z
    th = 1e-6
    R = np.array([[math.cos(th), -math.sin(th), 0.0],
                  [math.sin(th),  math.cos(th), 0.0],
                  [0.0,           0.0,          1.0]])
    T_curr = np.eye(4)
    T_des = np.eye(4)
    T_des[:3, :3] = R
    e_small = U.pose_error(T_curr, T_des, mode="small")
    e_so3 = U.pose_error(T_curr, T_des, mode="so3")
    # Position components are zero; orientation nearly equal
    assert np.allclose(e_small[:3], 0.0, atol=1e-12)
    assert np.allclose(e_so3[:3], 0.0, atol=1e-12)
    assert np.allclose(e_small[3:], e_so3[3:], atol=1e-9)


def test_dls_step_dimension_and_basic_behavior():
    """dls_step should return a vector of length n and reduce residual in a simple case."""
    m, n = 6, 3
    np.random.seed(0)
    J = np.random.randn(m, n)
    q = np.zeros(n)
    e = np.random.randn(m)
    lam = 1e-3
    dq = U.dls_step(J, e, lam)
    assert dq.shape == (n,)
    # One step should not increase ||e - J dq|| in least-squares sense
    new_res = np.linalg.norm(e - J @ dq)
    assert new_res <= np.linalg.norm(e) + 1e-10


# ------------------------------ Jacobian ----------------------------------

def test_jacobian_shapes_planar2r():
    chain = _planar_2r_chain()
    q = np.array([0.3, -0.2])
    J = chain.jacobian_space(q)
    assert J.shape == (6, 2)
    Jb = chain.jacobian_body(q)
    assert Jb.shape == (6, 2)


def test_manipulability_metrics_detects_singularity():
    chain = _planar_2r_chain()
    # Fully stretched configuration tends to be near singular for planar 2R
    q = np.array([0.0, 0.0])
    J = chain.jacobian_space(q)
    m = U.manipulability_metrics(J)
    assert set(m.keys()) == {"min_sing", "cond", "detJJT", "rank"}
    # Expect rank < 2 in planar sense; but geometric J is 6x2; check sigma_min small
    assert m["min_sing"] >= 0.0
    assert m["cond"] >= 1.0 or np.isinf(m["cond"])
    # At a generic non-singular configuration, rank should be at least 2 in the planar subspace
    q2 = np.array([0.7, -0.9])
    m2 = U.manipulability_metrics(chain.jacobian_space(q2))
    assert m2["min_sing"] > m["min_sing"]


# ------------------------------ Iterative IK ------------------------------

@pytest.mark.parametrize("target", [(1.2, 0.3), (0.5, 1.0), (0.0, 1.8)])
def test_iterative_converges_on_planar2r(target):
    """Iterative IK should converge to a reachable planar target from a reasonable seed."""
    chain = _planar_2r_chain(l1=1.0, l2=1.0)
    x, y = target
    # Ensure target is within reachable workspace (<= l1 + l2 - margin)
    r = math.hypot(x, y)
    assert r <= 2.0 + 1e-9

    solver = IterativeIK(lambda_damp=1e-3, tol=1e-9, itmax=200)
    q0 = np.array([0.1, -0.1])
    sols = solver.solve(chain, _T_xy(x, y), q0)
    assert isinstance(sols, list) and len(sols) >= 1
    q = sols[0]
    assert q.shape == (2,)

    # Validate by FK: end-effector position close to target
    T = chain.fkine(q).as_matrix()
    pos_err = np.linalg.norm(T[:3, 3][:2] - np.array([x, y]))
    assert pos_err < 1e-5


def test_iterative_respects_itmax_when_far_target():
    """If the target is far and tol is tiny with small itmax, solver should return last iterate."""
    chain = _planar_2r_chain(l1=1.0, l2=1.0)
    # Still reachable, but demand tiny tol and very few iterations
    x, y = 1.8, 0.0
    solver = IterativeIK(lambda_damp=1e-6, tol=1e-14, itmax=1)  # effectively 1 update
    q0 = np.array([0.0, 0.0])
    sols = solver.solve(chain, _T_xy(x, y), q0)
    q = sols[0]
    T = chain.fkine(q).as_matrix()
    # Residual likely > tol because we permitted only one iteration
    res = np.linalg.norm(U.pose_error(T, _T_xy(x, y)))
    assert res > solver.tol


def test_iterative_handles_multiple_random_targets():
    """Converge from random seeds to random reachable points."""
    rng = np.random.default_rng(2)
    chain = _planar_2r_chain(1.0, 1.0)
    solver = IterativeIK(lambda_damp=1e-3, tol=1e-8, itmax=300)
    for _ in range(5):
        # Sample reachable points inside the disk of radius 1.8
        r = 1.8 * math.sqrt(rng.random())
        phi = 2 * math.pi * rng.random()
        x, y = r * math.cos(phi), r * math.sin(phi)
        q0 = rng.uniform(-np.pi, np.pi, size=2)
        q = solver.solve(chain, _T_xy(x, y), q0)[0]
        T = chain.fkine(q).as_matrix()
        assert np.linalg.norm(T[:3, 3][:2] - np.array([x, y])) < 1e-4


# ------------------------------ Consistency -------------------------------

def test_fk_jacobian_consistency_numeric_directional_derivative():
    """
    Check that J maps joint velocities to end-effector twist approximately:
      xdot ≈ J(q) * qdot  (compare translation part over a small dt)
    """
    chain = _planar_2r_chain()
    q = np.array([0.7, -0.3])
    J = chain.jacobian_space(q)
    qdot = np.array([0.2, -0.4])
    dt = 1e-6

    T0 = chain.fkine(q).as_matrix()
    T1 = chain.fkine(q + qdot * dt).as_matrix()

    dp_num = (T1[:3, 3] - T0[:3, 3]) / dt
    v_pred = (J @ qdot)[3:]  # translational part in geometric Jacobian
    assert np.allclose(dp_num, v_pred, atol=1e-3)  # loose due to numerical diff
