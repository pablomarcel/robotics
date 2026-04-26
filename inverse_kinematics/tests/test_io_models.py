# inverse_kinematics/tests/test_io_models.py
# Tests for schemas, loaders, and builders in inverse_kinematics.io

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inverse_kinematics import io as IO


# ---------------------------- helpers ----------------------------

def _write_json(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return p


# ----------------------- schema & validation ---------------------

def test_problem_schema_validate_ok():
    prob = {
        "model": {"kind": "planar2r", "l1": 1.0, "l2": 0.8},
        "method": {"method": "analytic"},
        "pose": {"x": 0.7, "y": 0.2},
    }
    # Should not raise
    IO.validate_problem(prob)


def test_problem_schema_validate_fail_missing_pose():
    prob = {
        "model": {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
        "method": {"method": "analytic"},
        # "pose" missing
    }
    with pytest.raises(Exception):
        IO.validate_problem(prob)


# --------------------------- builders ----------------------------

def test_build_chain_high_level_planar2r():
    model = {"kind": "planar2r", "l1": 1.2, "l2": 0.7, "name": "p2r"}
    chain = IO.build_chain_from_model(model, validate=True)
    assert chain.n() == 2
    assert chain.name == "p2r"


def test_build_chain_high_level_wrist_and_sixdof():
    wrist = IO.build_chain_from_model({"kind": "spherical_wrist", "wrist_type": 1, "d_tool": 0.1}, validate=True)
    assert wrist.n() == 3

    six = IO.build_chain_from_model(
        {"kind": "six_dof_spherical", "l1": 0.5, "l2": 0.4, "wrist_type": 2, "d_tool": 0.05},
        validate=True,
    )
    assert six.n() == 6


def test_build_chain_low_level_dh_minimal():
    model = {
        "format": "dh",
        "links": [
            {"a": 0.0, "alpha": 0.0, "d": 0.0, "joint_type": "R"},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "joint_type": "R"},
        ],
        "M": np.eye(4).tolist(),
        "name": "low_dh",
    }
    IO.validate_chain_model(model)
    chain = IO.build_chain_from_model(model, validate=False)
    assert chain.n() == 2
    assert chain.name == "low_dh"


def test_build_chain_low_level_invalid_joint_type():
    bad = {
        "format": "dh",
        "links": [{"a": 0.0, "alpha": 0.0, "d": 0.0, "joint_type": "X"}],
    }
    # Schema may allow the key but builder should reject invalid enum
    with pytest.raises(Exception):
        IO.build_chain_from_model(bad, validate=False)


def test_build_chain_low_level_invalid_M_shape():
    bad = {
        "format": "dh",
        "links": [{"a": 0.0, "alpha": 0.0, "d": 0.0, "joint_type": "R"}],
        "M": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # not 4x4
    }
    with pytest.raises(Exception):
        IO.build_chain_from_model(bad, validate=False)


# --------------------------- I/O roundtrip -----------------------

def test_problem_roundtrip_json(tmp_path: Path):
    prob = {
        "model": {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
        "method": {"method": "iterative", "tol": 1e-8, "itmax": 200, "lambda": 1e-3},
        "pose": {"x": 0.9, "y": -0.1},
    }
    p = _write_json(tmp_path / "ik.json", prob)

    loaded = IO.load_problem_from_file(p, validate=True)
    assert isinstance(loaded, dict)
    # Save again and read back; keys should persist
    out = tmp_path / "ik_copy.json"
    IO.save_problem_json(out, loaded)
    loaded2 = json.loads(out.read_text(encoding="utf-8"))
    assert loaded2["model"]["kind"] == "planar2r"
    assert "tol" in loaded2["method"]
    assert set(loaded2.keys()) == {"model", "method", "pose"}


def test_save_solutions_json(tmp_path: Path):
    sols = [np.array([0.1, 0.2]), np.array([-0.3, 1.1])]
    out = tmp_path / "solutions.json"
    IO.save_solutions_json(out, sols)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "solutions" in data and isinstance(data["solutions"], list)
    assert data["solutions"][0] == [0.1, 0.2]
