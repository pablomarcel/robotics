# applied/tests/test_cli_design_and_diagram.py
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from applied import cli


def _run_cli(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main(args)
    return buf.getvalue()


def test_cli_design_list():
    out = _run_cli(["design", "--list"])
    # at least one preset key must be present
    assert "pendulum_" in out or "planar2r_" in out


def test_cli_design_preset_summary(tmp_path):
    out_file = tmp_path / "model.json"
    out = _run_cli(["design", "--preset", "planar2r_num", "--export", str(out_file)])
    assert "Model:" in out and "Planar2R" in out
    assert out_file.exists()


def test_cli_diagram_dot_and_json(tmp_path):
    # DOT
    out = _run_cli(["diagram", "dot", "--outdir", str(tmp_path), "--out", "c.dot"])
    assert str(tmp_path / "c.dot") in out
    # JSON
    out = _run_cli(["diagram", "json", "--outdir", str(tmp_path), "--out", "c.json"])
    assert str(tmp_path / "c.json") in out
