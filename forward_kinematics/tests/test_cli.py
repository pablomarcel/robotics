# forward_kinematics/tests/test_cli.py
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from forward_kinematics.cli import cli


# ------------------------------ helpers ------------------------------ #

def _write_json(p: Path, obj: dict) -> Path:
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return p

def _write_yaml(p: Path, obj: dict) -> Path:
    yaml = pytest.importorskip("yaml")
    p.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")
    return p

def _read_json_matrix(p: Path):
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "matrix" in data:
        return np.array(data["matrix"], dtype=float)
    return np.array(data, dtype=float)

def _simple_2r_spec_json():
    # Standard DH: L1 (a=0), L2 (a=1). Tool offset M translates +x by 1.
    # FK at q=[0,0] should yield x=2, y=0.
    return {
        "name": "planar_2r_test",
        "format": "dh",
        "links": [
            {"a": 0.0, "alpha": 0.0, "d": 0.0, "joint_type": "R"},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "joint_type": "R"},
        ],
        "M": [[1,0,0,1.0],
              [0,1,0,0.0],
              [0,0,1,0.0],
              [0,0,0,1.0]]
    }


# ------------------------------- tests ------------------------------- #

def test_validate_json_ok(tmp_path: Path):
    spec = _simple_2r_spec_json()
    spec_path = _write_json(tmp_path / "robot.json", spec)
    r = CliRunner().invoke(cli, ["validate", str(spec_path)])
    assert r.exit_code == 0, r.output
    assert "VALID" in r.output

def test_validate_yaml_ok(tmp_path: Path):
    # only runs if PyYAML is installed
    spec = _simple_2r_spec_json()
    spec_path = _write_yaml(tmp_path / "robot.yaml", spec)
    r = CliRunner().invoke(cli, ["validate", str(spec_path)])
    assert r.exit_code == 0, r.output
    assert "VALID" in r.output

def test_schema_export(tmp_path: Path):
    out = tmp_path / "robot.schema.json"
    r = CliRunner().invoke(cli, ["schema", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data.get("title") == "Forward Kinematics Robot Specification"

def test_fk_command_outputs_transform(tmp_path: Path):
    spec = _simple_2r_spec_json()
    spec_path = _write_json(tmp_path / "robot.json", spec)
    out = tmp_path / "T.json"
    # NOTE: --q is repeatable now
    r = CliRunner().invoke(cli, ["fk", str(spec_path), "--q", "0.0", "--q", "0.0", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists()
    T = _read_json_matrix(out)
    assert T.shape == (4, 4)
    assert np.allclose(T, np.array([[1,0,0,2.0],
                                    [0,1,0,0.0],
                                    [0,0,1,0.0],
                                    [0,0,0,1.0]]), atol=1e-9)

def test_jacobian_space_command(tmp_path: Path):
    spec = _simple_2r_spec_json()
    spec_path = _write_json(tmp_path / "robot.json", spec)
    out = tmp_path / "Js.json"
    r = CliRunner().invoke(cli, ["jacobian-space", str(spec_path), "--q", "0.0", "--q", "0.0", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists()
    J = _read_json_matrix(out)
    assert J.shape == (6, 2)

def test_jacobian_body_command(tmp_path: Path):
    spec = _simple_2r_spec_json()
    spec_path = _write_json(tmp_path / "robot.json", spec)
    out = tmp_path / "Jb.json"
    r = CliRunner().invoke(cli, ["jacobian-body", str(spec_path), "--q", "0.0", "--q", "0.0", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists()
    J = _read_json_matrix(out)
    assert J.shape == (6, 2)

def test_preset_scara(tmp_path: Path):
    T_out = tmp_path / "scara_T.json"
    r = CliRunner().invoke(
        cli,
        [
            "preset-scara",
            "--l1", "0.7", "--l2", "0.6", "--d", "0.18",
            "--q", "0.1", "--q", "0.2", "--q", "0.05",
            "-o", str(T_out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert T_out.exists()
    T = _read_json_matrix(T_out)
    assert T.shape == (4, 4)
    J_out = T_out.with_name(T_out.stem + "_J_space.json")
    assert J_out.exists()
    J = _read_json_matrix(J_out)
    assert J.shape[0] == 6

def test_preset_wrist(tmp_path: Path):
    T_out = tmp_path / "wrist_T.json"
    r = CliRunner().invoke(
        cli,
        [
            "preset-wrist",
            "--type", "1", "--d7", "0.1",
            "--q", "0.1", "--q", "0.2", "--q", "0.3",
            "-o", str(T_out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert T_out.exists()
    T = _read_json_matrix(T_out)
    assert T.shape == (4, 4)
    J_out = T_out.with_name(T_out.stem + "_J_body.json")
    assert J_out.exists()
    J = _read_json_matrix(J_out)
    assert J.shape[0] == 6

def test_diagram_dot_export(tmp_path: Path):
    out = tmp_path / "classes.dot"
    r = CliRunner().invoke(cli, ["diagram-dot", "-o", str(out)])
    assert r.exit_code == 0, r.output
    text = out.read_text(encoding="utf-8")
    assert "digraph ForwardDiagram" in text
    assert "forward_kinematics.core" in text  # cluster label present

def test_sphinx_skeleton(tmp_path: Path):
    dest = tmp_path / "docs"
    r = CliRunner().invoke(cli, ["sphinx-skel", str(dest)])
    assert r.exit_code == 0, r.output
    assert (dest / "conf.py").exists()
    assert (dest / "index.rst").exists()
    assert (dest / "api.rst").exists()
    assert (dest / "Makefile").exists()
