# velocity_kinematics/tests/test_app.py
"""
App bootstrap & CLI integration tests.

Covers:
- velocity_kinematics.app.main() round-trips through CLI 'fk' subcommand
- Ensures app creates default workdirs (velocity_kinematics/in, velocity_kinematics/out)
- Diagram subcommand behavior (success when pylint present; clean API error otherwise)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from velocity_kinematics import app as app_mod


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


def _read_stdout_json(capsys) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "expected JSON on stdout"
    return json.loads(out)


def test_app_main_fk_smoke(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    spec = _write_planar_2r_json(tmp_path)
    # Run through the actual app entrypoint (argparser + dispatch)
    code = app_mod.main(["fk", str(spec), "--q", "0.1,0.2"])
    assert code == 0
    payload = _read_stdout_json(capsys)
    assert "T_0e" in payload and "frames" in payload
    T = np.asarray(payload["T_0e"], dtype=float)
    assert T.shape == (4, 4)


def test_app_ensures_default_workdirs_exist() -> None:
    cfg = app_mod.AppConfig.default()
    # The app creates these on startup; call ensure_workdirs directly to avoid CLI noise.
    app_mod.ensure_workdirs(cfg)
    assert cfg.indir.exists() and cfg.indir.is_dir()
    assert cfg.outdir.exists() and cfg.outdir.is_dir()


def test_app_diagram_subcommand_handles_optional_pylint(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    If pylint is installed, diagram should succeed (exit 0).
    If not, the API raises a clear error and app returns exit code 2.
    """
    has_pylint = importlib.util.find_spec("pylint.pyreverse.main") is not None
    # Use explicit outdir to keep repo tree clean
    args = ["diagram", "--out", str(tmp_path)]
    code = app_mod.main(args)
    if has_pylint:
        assert code == 0
        out = capsys.readouterr().out.strip()
        # Expect a JSON dict with the diagram path_planning
        assert out.startswith("{") and out.endswith("}")
    else:
        # API error path_planning uses exit code 2 in cli layer
        assert code == 2
