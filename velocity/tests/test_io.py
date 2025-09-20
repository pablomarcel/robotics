# velocity/tests/test_io.py
"""
I/O tests for the Velocity Kinematics Toolkit.

Covers:
- load_dh_from_file: happy path (JSON), schema validation errors
- Optional YAML path (skipped if PyYAML not installed)
- 'tool' field validation for both xyz and 4x4 matrix forms
- load_urdf_from_file: graceful fallback when urdfpy isn't installed
- save_report: NumPy-safe JSON write

We avoid requiring external parsers; YAML and URDF tests are skipped or
use minimal content to keep the suite fast and robust.
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from velocity.io import load_dh_from_file, load_urdf_from_file, save_report


# ----------------------------- Helpers --------------------------------------- #

def _dh_minimal_dict() -> dict:
    return {
        "name": "planar2r",
        "joints": [
            {"name": "j1", "type": "R", "alpha": 0.0, "a": 1.0, "d": 0.0, "theta": 0.0},
            {"name": "j2", "type": "R", "alpha": 0.0, "a": 1.0, "d": 0.0, "theta": 0.0},
        ],
        "tool": {"xyz": [0.0, 0.0, 0.0]},
    }


# ----------------------------- JSON happy path -------------------------------- #

def test_load_dh_from_json_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "arm.json"
    p.write_text(json.dumps(_dh_minimal_dict(), indent=2), encoding="utf-8")

    data = load_dh_from_file(p)
    assert isinstance(data, dict)
    assert "joints" in data and len(data["joints"]) == 2
    assert data["joints"][0]["type"] == "R"
    assert data["tool"]["xyz"] == [0.0, 0.0, 0.0]


# ----------------------------- YAML path (optional) --------------------------- #

@pytest.mark.skipif(importlib.util.find_spec("yaml") is None, reason="PyYAML not installed")
def test_load_dh_from_yaml_when_available(tmp_path: Path) -> None:
    p = tmp_path / "arm.yml"
    yaml_text = (
        "name: planar2r\n"
        "joints:\n"
        "  - {name: j1, type: R, alpha: 0.0, a: 1.0, d: 0.0, theta: 0.0}\n"
        "  - {name: j2, type: R, alpha: 0.0, a: 1.0, d: 0.0, theta: 0.0}\n"
        "tool: {xyz: [0.0, 0.0, 0.0]}\n"
    )
    p.write_text(yaml_text, encoding="utf-8")

    data = load_dh_from_file(p)
    assert data["name"] == "planar2r"
    assert len(data["joints"]) == 2


# --------------------------- Schema validation errors ------------------------- #

def test_load_dh_schema_errors(tmp_path: Path) -> None:
    # Missing joints key
    p1 = tmp_path / "bad1.json"
    p1.write_text(json.dumps({"name": "no_joints"}, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dh_from_file(p1)

    # Bad joint type
    bad = _dh_minimal_dict()
    bad["joints"][0]["type"] = "X"
    p2 = tmp_path / "bad2.json"
    p2.write_text(json.dumps(bad, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dh_from_file(p2)

    # Non-numeric alpha
    bad = _dh_minimal_dict()
    bad["joints"][1]["alpha"] = "oops"
    p3 = tmp_path / "bad3.json"
    p3.write_text(json.dumps(bad, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dh_from_file(p3)


def test_tool_field_validation(tmp_path: Path) -> None:
    # Valid 4x4 tool matrix
    good = _dh_minimal_dict()
    good["tool"] = np.eye(4).tolist()
    p = tmp_path / "good_tool_mat.json"
    p.write_text(json.dumps(good, indent=2), encoding="utf-8")
    data = load_dh_from_file(p)
    assert isinstance(data, dict) and "joints" in data

    # Invalid shaped matrix
    bad = _dh_minimal_dict()
    bad["tool"] = [[1, 0], [0, 1]]
    p_bad = tmp_path / "bad_tool_mat.json"
    p_bad.write_text(json.dumps(bad, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dh_from_file(p_bad)

    # Invalid xyz vector length
    bad = _dh_minimal_dict()
    bad["tool"] = {"xyz": [0.0, 0.0]}  # too short
    p_bad2 = tmp_path / "bad_tool_xyz.json"
    p_bad2.write_text(json.dumps(bad, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dh_from_file(p_bad2)


# --------------------------- URDF loader fallback ----------------------------- #

@pytest.mark.skipif(importlib.util.find_spec("urdfpy") is not None, reason="urdfpy installed; fallback test expects absence")
def test_load_urdf_returns_raw_when_no_urdfpy(tmp_path: Path) -> None:
    urdf_text = """<?xml version="1.0"?>
<robot name="toy">
  <link name="base"/>
</robot>
"""
    p = tmp_path / "toy.urdf"
    p.write_text(urdf_text, encoding="utf-8")

    data = load_urdf_from_file(p)
    # Without urdfpy, function returns {"name": stem, "raw": text}
    assert "raw" in data
    assert data["name"] == "toy"
    assert "link" in data["raw"]


# --------------------------- save_report JSON writer -------------------------- #

def test_save_report_numpy_safe(tmp_path: Path) -> None:
    payload = {
        "array": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "scalar": np.float64(3.14),
        "int": np.int64(7),
        "nested": {"v": np.array([1, 2, 3])},
    }
    out = tmp_path / "report.json"
    save_report(out, payload)

    # Reload and check plain JSON types
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["array"] == [[1.0, 2.0], [3.0, 4.0]]
    assert isinstance(loaded["scalar"], float)
    assert isinstance(loaded["int"], float)  # coerced to float by default()
    assert loaded["nested"]["v"] == [1, 2, 3]
