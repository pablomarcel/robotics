# time/tests/test_cli_smoke.py
import pytest
from click.testing import CliRunner
from time_optimal_control.cli import cli

def test_diagram_cli_smoke(tmp_path):
    out = tmp_path / "diagram.puml"
    r = CliRunner().invoke(cli, ["diagram", "--out", str(out)])
    assert r.exit_code == 0
    assert out.exists()
