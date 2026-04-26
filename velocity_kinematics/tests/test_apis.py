# velocity_kinematics/tests/test_apis.py
"""
VelocityAPI façade tests.

Covers:
- load_robot (DH JSON) -> RobotSpec
- fk, jacobian_geometric, jacobian_analytic via API
- resolved_rates and newton_ik via API
- lu_solve / lu_inverse paths by monkeypatching a tiny 'velocity_kinematics.tools.lu' module

We avoid optional dependencies (pylint/pyreverse) here.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from velocity_kinematics.apis import VelocityAPI, RobotFormat, APIError


def _write_planar_2r_json(tmp: Path, l1: float = 1.0, l2: float = 1.0) -> Path:
    data = {
        "name": "planar2r",
        "joints": [
            {"name": "j1", "type": "R", "alpha": 0.0, "a": float(l1), "d": 0.0, "theta": 0.0},
            {"name": "j2", "type": "R", "alpha": 0.0, "a": float(l2), "d": 0.0, "theta": 0.0},
        ],
        "tool": {"xyz": [0.0, 0.0, 0.0]},
    }
    p = tmp / "arm.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Loading and basic kinematics
# --------------------------------------------------------------------------- #

def test_api_load_fk_jacobians(tmp_path: Path) -> None:
    api = VelocityAPI(default_in=tmp_path, default_out=tmp_path)
    spec_path = _write_planar_2r_json(tmp_path)
    spec = api.load_robot(spec_path)

    assert spec.fmt is RobotFormat.DH
    assert spec.path and spec.path.exists()
    assert spec.name == "planar2r"

    q = [0.2, -0.3]
    fk = api.fk(spec, q)
    Jg = api.jacobian_geometric(spec, q)
    JA = api.jacobian_analytic(spec, q, euler="ZYX")

    assert "T_0e" in fk and "frames" in fk
    assert np.asarray(fk["T_0e"]).shape == (4, 4)
    assert Jg.shape == (6, 2)
    assert JA.shape == (6, 2)
    # last row (about z) is ones for planar 2R
    assert np.allclose(Jg[5, :], [1.0, 1.0])


def test_api_resolved_rates_and_newton_ik(tmp_path: Path) -> None:
    api = VelocityAPI(default_in=tmp_path, default_out=tmp_path)
    spec = api.load_robot(_write_planar_2r_json(tmp_path))

    q = [0.4, -0.2]
    xdot = [0.1, 0.0, 0.0, 0, 0, 0]
    qdot = api.resolved_rates(spec, q, xdot, damping=1e-6)
    assert qdot.shape == (2,)
    assert np.linalg.norm(qdot) > 0

    # Position-only IK target
    q0 = [0.1, 0.1]
    x_target = {"p": [1.4, 0.2, 0.0]}
    q_sol, info = api.newton_ik(spec, q0, x_target, max_iter=100, tol=1e-9)
    assert info["converged"] is True
    T = api.fk(spec, q_sol)["T_0e"]
    err = np.linalg.norm(np.asarray(T)[:3, 3] - np.array(x_target["p"]))
    assert err < 1e-6


# --------------------------------------------------------------------------- #
# LU tool path via monkeypatching a fake velocity_kinematics.tools.lu
# --------------------------------------------------------------------------- #

def test_api_lu_tools_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Provide a tiny 'velocity_kinematics.tools.lu' module in sys.modules so VelocityAPI.lu_*
    methods work without a real tools package yet.
    """
    mod = types.ModuleType("velocity_kinematics.tools.lu")

    def lu_factor(A):
        A = np.asarray(A, dtype=float)
        # naive Doolittle without pivoting (sufficient for our test matrices)
        n = A.shape[0]
        L = np.eye(n)
        U = A.copy()
        for i in range(n - 1):
            pivot = U[i, i]
            for j in range(i + 1, n):
                m = U[j, i] / pivot
                L[j, i] = m
                U[j, i:] = U[j, i:] - m * U[i, i:]
        return L, U

    def lu_solve(L, U, b):
        L = np.asarray(L, float)
        U = np.asarray(U, float)
        b = np.asarray(b, float).reshape(-1)
        # forward_kinematics
        y = np.zeros_like(b)
        for i in range(L.shape[0]):
            y[i] = b[i] - L[i, :i] @ y[:i]
        # back
        x = np.zeros_like(b)
        for i in reversed(range(U.shape[0])):
            x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
        return x

    def lu_inverse(L, U):
        n = L.shape[0]
        I = np.eye(n)
        cols = [lu_solve(L, U, I[:, i]) for i in range(n)]
        return np.column_stack(cols)

    mod.lu_factor = lu_factor
    mod.lu_solve = lu_solve
    mod.lu_inverse = lu_inverse

    # Register hierarchy packages if missing
    if "velocity_kinematics.tools" not in sys.modules:
        sys.modules["velocity_kinematics.tools"] = types.ModuleType("velocity_kinematics.tools")
    sys.modules["velocity_kinematics.tools.lu"] = mod

    api = VelocityAPI()
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    x = api.lu_solve(A, b)
    assert np.allclose(x, np.linalg.solve(A, b))

    Ainv = api.lu_inverse(A)
    assert np.allclose(Ainv @ A, np.eye(2), atol=1e-12)


# --------------------------------------------------------------------------- #
# Error path coverage
# --------------------------------------------------------------------------- #

def test_api_load_robot_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "robot.txt"
    p.write_text("oops", encoding="utf-8")
    api = VelocityAPI()
    with pytest.raises(APIError):
        api.load_robot(p)
