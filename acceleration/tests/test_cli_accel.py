from __future__ import annotations
import json
from pathlib import Path
import pytest
from click.testing import CliRunner

import acceleration.cli as cli_mod

def test_cli_help_runs():
    runner = CliRunner()
    res = runner.invoke(cli_mod.cli, ["--help"])
    assert res.exit_code == 0
    assert "acceleration" in res.output.lower() or "help" in res.output.lower()

def test_cli_diagram_mermaid(tmp_path):
    # If your CLI exposes a diagram command mirroring inverse/cli.py
    runner = CliRunner()
    out = tmp_path / "diagram.md"
    res = runner.invoke(cli_mod.cli, ["diagram-mermaid", "--out", str(out)])
    # If command not present, this will fail; adjust if your command name differs.
    assert res.exit_code == 0
    txt = out.read_text(encoding="utf-8")
    assert "```mermaid" in txt and "classDiagram" in txt
