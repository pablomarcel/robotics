# acceleration/tests/test_2r_forward_inverse.py
from __future__ import annotations

import numpy as np
import pytest

from acceleration.backends.base import ChainState
from acceleration.backends.numpy_backend import Planar2R
from acceleration.utils import jdot_qdot_fd


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_shapes_and_frames():
    be = Planar2R(l1=0.7, l2=0.9)
    assert be.dof() == 2
    frames = tuple(be.frames())
    assert frames == ("ee",)

    q = np.array([0.1, -0.2])
    J = be.jacobian("ee", q)
    assert J.shape == (2, 2)
    assert J.dtype == float

    qd = np.array([0.3, -0.4])
    b = be.jdot_qdot("ee", q, qd)
    assert b.shape == (2,)
    assert b.dtype == float

    st = ChainState(q=q, qd=qd, qdd=np.array([0.5, -0.6]))
    a = be.spatial_accel("ee", st)
    assert a.shape == (2,)
    assert a.dtype == float


def test_jdot_qdot_matches_central_fd(tol: float = 1e-9):
    """
    Validate the closed-form bias term with a directional central finite-difference:
        (J̇ q̇) ≈ [(J(q+ε q̇) - J(q-ε q̇)) / (2ε)] @ q̇
    """
    be = Planar2R(l1=0.7, l2=0.9, _fd_check=False)
    rng = _rng(123)

    for _ in range(10):
        q = rng.uniform(low=-1.0, high=1.0, size=2)
        qd = rng.normal(size=2)
        closed = be.jdot_qdot("ee", q, qd)
        fd = jdot_qdot_fd(lambda x: be.jacobian("ee", x), q, qd)
        assert np.allclose(closed, fd, atol=tol, rtol=tol), f"\nclosed={closed}\nfd={fd}"


def test_spatial_accel_equals_J_qdd_plus_bias():
    be = Planar2R(l1=1.0, l2=0.5)
    rng = _rng(7)

    for _ in range(8):
        q = rng.uniform(low=-0.9, high=0.9, size=2)
        qd = rng.normal(size=2) * 0.6
        qdd = rng.normal(size=2) * 0.4

        st = ChainState(q=q, qd=qd, qdd=qdd)

        J = be.jacobian("ee", q)
        bias = be.jdot_qdot("ee", q, qd)
        expected = J @ qdd + bias

        a = be.spatial_accel("ee", st)
        assert np.allclose(a, expected, atol=1e-12)


def test_zero_velocity_zero_bias():
    be = Planar2R(l1=0.8, l2=0.6)

    q = np.array([0.2, -0.4])
    qd = np.zeros(2)
    qdd = np.array([0.1, -0.2])

    bias = be.jdot_qdot("ee", q, qd)
    assert np.allclose(bias, np.zeros(2), atol=1e-12)

    st = ChainState(q=q, qd=qd, qdd=qdd)
    a = be.spatial_accel("ee", st)
    assert np.allclose(a, be.jacobian("ee", q) @ qdd, atol=1e-12)


def test_bias_is_quadratic_in_velocities():
    """
    For the Planar-2R closed form, (J̇ q̇) is quadratic in q̇.
    Check scaling: q̇ -> k q̇  ⇒  bias -> k^2 * bias.
    """
    be = Planar2R(l1=0.9, l2=0.7)
    q = np.array([0.3, -0.2])
    qd = np.array([0.4, -0.5])

    base = be.jdot_qdot("ee", q, qd)
    for k in [0.5, 2.0, -1.5]:
        scaled = be.jdot_qdot("ee", q, k * qd)
        assert np.allclose(scaled, (k ** 2) * base, atol=1e-12)
