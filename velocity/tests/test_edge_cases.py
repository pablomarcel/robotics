# velocity/tests/test_edge_cases.py
"""
Edge-case tests for Velocity Kinematics.

Covers:
- Analytic Jacobian fallback near Euler-map singularity (ZXZ @ identity)
- DHRobot.from_spec accepts a raw dict (not only RobotSpec)
- Spherical wrist zero-block is FALSE when the TCP is offset from the wrist center
- @timed decorator attaches a runtime attribute
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from velocity import core, design, utils


# --------------------------------------------------------------------------- #
# Analytic Jacobian fallback: ZXZ singularity (beta ~ 0) → return geometric J
# --------------------------------------------------------------------------- #

def test_analytic_fallback_near_zxz_singularity():
    # Type-1 wrist (~ Z–X–Z). At q = [0,0,0], R = I so ZXZ has beta = 0 → singular.
    wrist = design.spherical_wrist(wrist_type=1, d_tool=0.0)
    q = np.zeros(3)
    Jg = wrist.jacobian_geometric(q)
    JA = wrist.jacobian_analytic(q, euler="ZXZ")
    # Implementation should gracefully fall back to geometric J
    assert np.allclose(JA, Jg, atol=1e-12)


# --------------------------------------------------------------------------- #
# DHRobot.from_spec accepts raw dicts (useful in tests/toy examples)
# --------------------------------------------------------------------------- #

def test_dhrobot_from_raw_dict():
    spec = {
        "name": "toy2r",
        "joints": [
            {"name": "j1", "type": "R", "alpha": 0.0, "a": 0.7, "d": 0.0, "theta": 0.0},
            {"name": "j2", "type": "R", "alpha": 0.0, "a": 0.5, "d": 0.0, "theta": 0.0},
        ],
        "tool": {"xyz": [0.0, 0.0, 0.0]},
    }
    robot = core.DHRobot.from_spec(spec)  # type: ignore[arg-type]
    q = np.array([0.2, -0.3])
    T = robot.fk(q)["T_0e"]
    assert T.shape == (4, 4)
    J = robot.jacobian_geometric(q)
    assert J.shape == (6, 2)


# --------------------------------------------------------------------------- #
# Spherical wrist zero-block should be FALSE when d_tool != 0
# --------------------------------------------------------------------------- #

def test_zero_block_false_when_tcp_offset_from_wrist_center():
    # 6R with nonzero tool offset: translational coupling from wrist rates appears
    robot = design.six_dof_spherical(l1=0.4, l2=0.3, wrist_type=1, d_tool=0.12)
    q = np.array([0.3, -0.4, 0.1, 0.2, -0.5, 0.4])
    assert design.is_spherical_wrist_zero_block(robot, q) is False


# --------------------------------------------------------------------------- #
# @timed decorator exposes last_runtime_s on the wrapper
# --------------------------------------------------------------------------- #

def test_timed_decorator_sets_runtime_attr():
    @utils.timed
    def f(n: int) -> int:
        s = 0
        for i in range(n):
            s += i
        return s

    out = f(1000)
    assert out == sum(range(1000))
    # The decorator should attach this attribute after the call
    assert getattr(f, "last_runtime_s", None) is not None
    assert isinstance(f.last_runtime_s, float)
    assert f.last_runtime_s >= 0.0
