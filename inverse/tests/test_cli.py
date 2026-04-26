# inverse/tests/test_cli.py
# Pytest suite for the inverse CLI.
# These tests exercise the happy-path flows and file I/O surfaces using Click's CliRunner.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

# Import the CLI entrypoint (Click group)
from inverse.cli import cli as inverse_cli


# --------------------------- helpers ---------------------------

def _write_json(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return p


def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _T_from_xy(x: float, y: float) -> list[list[float]]:
    """Planar pose with identity rotation_kinematics and translation (x,y,0)."""
    T = np.eye(4)
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    return T.tolist()


# --------------------------- tests -----------------------------

def test_problem_validate_ok(tmp_path: Path):
    """problem-validate: accept a well-formed IK problem."""
    problem = {
        "model": {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
        "method": {"method": "analytic"},
        "pose": {"x": 1.0, "y": 1.0},
    }
    p = _write_json(tmp_path / "ok_problem.json", problem)

    runner = CliRunner()
    res = runner.invoke(inverse_cli, ["problem-validate", str(p)])
    assert res.exit_code == 0, res.output
    assert "VALID" in res.output


def test_problem_validate_fail(tmp_path: Path):
    """problem-validate: reject malformed input."""
    bad = {
        "model": {"kind": "planar2r", "l1": 1.0},  # missing l2
        "method": {"method": "analytic"},
        "pose": {"x": 1.0, "y": 1.0},
    }
    p = _write_json(tmp_path / "bad_problem.json", bad)

    runner = CliRunner()
    res = runner.invoke(inverse_cli, ["problem-validate", str(p)])
    assert res.exit_code != 0
    assert "INVALID" in res.output


def test_ik_planar2r_analytic_writes_solutions(tmp_path: Path):
    """ik-planar2r (analytic): writes two branches for a reachable target."""
    out = tmp_path / "solutions.json"
    runner = CliRunner()
    res = runner.invoke(
        inverse_cli,
        [
            "ik-planar2r",
            "--l1", "1.0",
            "--l2", "1.0",
            "--x", "1.0",
            "--y", "1.0",
            "--method", "analytic",
            "-o", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    data = _read_json(out)
    sols = data["solutions"]
    assert isinstance(sols, list) and len(sols) >= 2
    # Each solution should be length-2 for planar 2R
    assert all(isinstance(s, list) and len(s) == 2 for s in sols)


def test_ik_solve_iterative_with_T_path(tmp_path: Path):
    """ik-solve (iterative) using a target 4x4 transform file."""
    Tfile = _write_json(tmp_path / "target_T.json", _T_from_xy(1.2, 0.3))
    out = tmp_path / "it.json"
    runner = CliRunner()
    res = runner.invoke(
        inverse_cli,
        [
            "ik-solve",
            "--model", "planar2r",
            "--l1", "1.0",
            "--l2", "1.0",
            "--method", "iterative",
            "--T-path", str(Tfile),
            "--q0", "0.1", "--q0", "0.1",
            "--tol", "1e-8",
            "--itmax", "200",
            "--lambda-damp", "1e-3",
            "-o", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    data = _read_json(out)
    sols = data["solutions"]
    assert len(sols) >= 1
    assert len(sols[0]) == 2  # 2R


def test_ik_batch_multiple_poses(tmp_path: Path):
    """ik-batch: solves a list of poses and returns list-of-lists of solutions."""
    poses = [
        {"x": 1.0, "y": 0.0},
        {"x": 0.5, "y": 1.0},
        {"T": _T_from_xy(0.8, 0.2)},
    ]
    poses_path = _write_json(tmp_path / "poses.json", poses)
    out = tmp_path / "batch.json"
    runner = CliRunner()
    res = runner.invoke(
        inverse_cli,
        [
            "ik-batch",
            "--model", "planar2r",
            "--l1", "1.0",
            "--l2", "1.0",
            "--poses", str(poses_path),
            "--method", "analytic",
            "-o", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    data = _read_json(out)
    sols_per_pose = data["solutions"]
    assert isinstance(sols_per_pose, list) and len(sols_per_pose) == len(poses)
    # Each pose should have at least one solution (or two)
    assert all(isinstance(slist, list) and len(slist) >= 1 for slist in sols_per_pose)


def test_problem_solve_end_to_end(tmp_path: Path):
    """problem-solve: end-to-end solve from a single JSON file."""
    problem = {
        "model": {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
        "method": {"method": "analytic"},
        "pose": {"x": 1.0, "y": 0.5},
    }
    p = _write_json(tmp_path / "prob.json", problem)
    out = tmp_path / "solutions.json"

    runner = CliRunner()
    res = runner.invoke(inverse_cli, ["problem-solve", str(p), "-o", str(out)])
    assert res.exit_code == 0, res.output
    sols = _read_json(out)["solutions"]
    assert isinstance(sols, list) and len(sols) >= 1
    assert len(sols[0]) == 2


def test_diagram_mermaid_outputs_markdown(tmp_path: Path):
    """diagram-mermaid: emits a Markdown file with Mermaid diagram code."""
    md = tmp_path / "classes.md"
    runner = CliRunner()
    res = runner.invoke(inverse_cli, ["diagram-mermaid", "-o", str(md)])
    assert res.exit_code == 0, res.output
    text = md.read_text(encoding="utf-8")
    assert "classDiagram" in text or "```mermaid" in text or len(text) > 0


def test_sphinx_skel_creates_minimal_docs(tmp_path: Path):
    """sphinx-skel: creates a minimal Sphinx tree that can be built later."""
    docs_dir = tmp_path / "docs"
    runner = CliRunner()
    res = runner.invoke(inverse_cli, ["sphinx-skel", str(docs_dir)])
    assert res.exit_code == 0, res.output
    # Core files exist
    assert (docs_dir / "conf.py").exists()
    assert (docs_dir / "index.rst").exists()
    assert (docs_dir / "api.rst").exists()
    assert (docs_dir / "Makefile").exists()
