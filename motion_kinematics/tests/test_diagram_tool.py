# motion_kinematics/tests/test_diagram_tool.py
import json
import os
import shutil
from pathlib import Path

import pytest

from motion_kinematics.tools.diagram import DiagramTool, DiagramConfig


def test_discover_and_export_json_snapshot(tmp_path):
    outdir = tmp_path / "out"
    tool = DiagramTool(DiagramConfig(package="motion_kinematics", out_dir=outdir))

    json_path = tool.export_model_json(out_file="classes.json")
    p = Path(json_path)
    assert p.exists() and p.is_file()

    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["package"] == "motion_kinematics"
    # sanity: we discovered some classes
    assert isinstance(data["classes"], list) and len(data["classes"]) > 0
    # spot-check core types
    qualnames = {c["qualname"] for c in data["classes"]}
    assert "motion_kinematics.core.SE3" in qualnames
    assert "motion_kinematics.core.Rotation" in qualnames


def test_emit_plantuml_and_mermaid(tmp_path):
    outdir = tmp_path / "out"
    tool = DiagramTool(DiagramConfig(out_dir=outdir))

    pu_path = tool.emit_plantuml(out_file="classes.puml")
    md_path = tool.emit_mermaid(out_file="classes.mmd")

    pu = Path(pu_path); md = Path(md_path)
    assert pu.exists() and md.exists()
    # Quick content checks
    txt_pu = pu.read_text(encoding="utf-8")
    txt_md = md.read_text(encoding="utf-8")
    assert "@startuml" in txt_pu and "@enduml" in txt_pu
    assert txt_md.splitlines()[0].strip().lower().startswith("classdiagram")


def test_render_graphviz_optional(tmp_path):
    """Run only if graphviz python package AND 'dot' executable are available."""
    try:
        import graphviz  # noqa: F401
    except Exception:
        pytest.skip("graphviz Python package not installed")
    if shutil.which("dot") is None:
        pytest.skip("'dot' (Graphviz CLI) not found on PATH")

    outdir = tmp_path / "out"
    tool = DiagramTool(DiagramConfig(out_dir=outdir))
    path = tool.render_graphviz(fmt="png", rankdir="LR", out_stem="classes")
    p = Path(path)
    assert p.exists() and p.suffix.lower() == ".png"


def test_render_pyreverse_optional(tmp_path):
    """Run only if pyreverse is on PATH; skip if it fails in this environment."""
    if shutil.which("pyreverse") is None:
        pytest.skip("pyreverse not on PATH")

    outdir = tmp_path / "out"
    tool = DiagramTool(DiagramConfig(out_dir=outdir))
    try:
        files = tool.render_pyreverse(fmt="png")
    except RuntimeError as e:
        pytest.skip(f"pyreverse failed to run here: {e}")
    # Should produce at least classes.png or packages.png
    assert isinstance(files, list) and len(files) >= 1
    for f in files:
        assert Path(f).exists()


def test_render_all_compiles_even_without_optional_backends(tmp_path):
    outdir = tmp_path / "out"
    tool = DiagramTool(DiagramConfig(out_dir=outdir))
    artifacts = tool.render_all()

    # Always present
    assert "json" in artifacts and isinstance(artifacts["json"], str)
    assert Path(artifacts["json"]).exists()

    assert "plantuml" in artifacts and isinstance(artifacts["plantuml"], str)
    assert Path(artifacts["plantuml"]).exists()

    assert "mermaid" in artifacts and isinstance(artifacts["mermaid"], str)
    assert Path(artifacts["mermaid"]).exists()

    # Optional backends may be unavailable; just ensure keys exist
    assert "graphviz" in artifacts
    assert "pyreverse" in artifacts
