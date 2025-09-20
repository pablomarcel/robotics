# velocity/tests/test_cli.py
"""
CLI smoke tests for the Velocity Kinematics Toolkit.

We exercise:
- fk
- jacobian
- resolved-rates
- newton-ik (position-only)
- lu-solve and lu-inv

Notes
-----
- We avoid optional dependencies (pylint/pyreverse, graphviz) here.
- Robot specs are written as JSON to skip the PyYAML dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from velocity.cli import VelocityCLI


def _write_planar_2r_json(tmp: Path, l1: float = 1.0, l2: float = 1.0) -> Path:
    robot = {
        "name": "planar2r",
        "joints": [
            {"name": "j1", "type": "R", "alpha": 0.0, "a": float(l1), "d": 0.0, "theta": 0.0},
            {"name": "j2", "type": "R", "alpha": 0.0, "a": float(l2), "d": 0.0, "theta": 0.0},
        ],
        "tool": {"xyz": [0.0, 0.0, 0.0]},
    }
    p = tmp / "planar2r.json"
    p.write_text(json.dumps(robot, indent=2), encoding="utf-8")
    return p


def _read_stdout_json(capsys) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "expected JSON on stdout"
    return json.loads(out)


# --------------------------------------------------------------------------- #
# fk
# --------------------------------------------------------------------------- #

def test_cli_fk_outputs_pose_and_frames(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    spec = _write_planar_2r_json(tmp_path)
    cli = VelocityCLI()
    code = cli.run(["fk", str(spec), "--q", "0.1,0.2"])
    assert code == 0
    payload = _read_stdout_json(capsys)
    assert "T_0e" in payload and "frames" in payload
    assert np.array(payload["T_0e"]).shape == (4, 4)
    assert isinstance(payload["frames"], list)
    assert len(payload["frames"]) == 3  # base + 2 links (tool = identity)


# --------------------------------------------------------------------------- #
# jacobian
# --------------------------------------------------------------------------- #

def test_cli_jacobian_returns_6xn(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    spec = _write_planar_2r_json(tmp_path)
    cli = VelocityCLI()
    code = cli.run(["jacobian", str(spec), "--q", "0.3,-0.1"])
    assert code == 0
    payload = _read_stdout_json(capsys)
    J = np.asarray(payload["J"], dtype=float)
    assert J.shape == (6, 2)
    # For planar 2R, lower z-rows should be ones in last row
    assert np.allclose(J[5, :], np.array([1.0, 1.0]))


# --------------------------------------------------------------------------- #
# resolved-rates
# --------------------------------------------------------------------------- #

def test_cli_resolved_rates_basic(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    spec = _write_planar_2r_json(tmp_path)
    cli = VelocityCLI()
    args = [
        "resolved-rates",
        str(spec),
        "--q", "0.4,-0.2",
        "--xdot", "0.1,0.0,0, 0,0,0",
        "--damping", "1e-6",
    ]
    code = cli.run(args)
    assert code == 0
    payload = _read_stdout_json(capsys)
    qdot = np.asarray(payload["qdot"], dtype=float)
    assert qdot.shape == (2,)
    # sanity: not all zeros
    assert np.linalg.norm(qdot) > 0.0


# --------------------------------------------------------------------------- #
# newton-ik (position only)
# --------------------------------------------------------------------------- #

def test_cli_newton_ik_position_only(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    spec = _write_planar_2r_json(tmp_path)
    cli = VelocityCLI()
    # reachable target
    code = cli.run([
        "newton-ik",
        str(spec),
        "--q0", "0.1,0.1",
        "--p", "1.4,0.2,0.0",
        "--max-iter", "80",
        "--tol", "1e-9",
    ])
    assert code == 0
    payload = _read_stdout_json(capsys)
    q = np.asarray(payload["q"], dtype=float)
    info = payload["info"]
    assert q.shape == (2,)
    assert info["converged"] is True


# --------------------------------------------------------------------------- #
# LU helpers
# --------------------------------------------------------------------------- #

def test_cli_lu_solve_and_inv(capsys: pytest.CaptureFixture) -> None:
    cli = VelocityCLI()

    # Solve A x = b
    A = "[[2,1],[1,3]]"
    b = "[1,2]"
    code = cli.run(["lu-solve", "--A", A, "--b", b])
    assert code == 0
    payload = _read_stdout_json(capsys)
    x = np.asarray(payload["x"], dtype=float)
    assert np.allclose(x, np.linalg.solve(np.array([[2,1],[1,3]], float), np.array([1,2], float)))

    # Inverse
    code = cli.run(["lu-inv", "--A", A])
    assert code == 0
    payload = _read_stdout_json(capsys)
    Ainv = np.asarray(payload["A_inv"], dtype=float)
    assert np.allclose(Ainv @ np.array([[2,1],[1,3]], float), np.eye(2), atol=1e-12)
