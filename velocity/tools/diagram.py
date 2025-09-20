# velocity/tests/diagram.py
"""
Diagram tests for the Velocity Kinematics module.

These tests validate that our diagram helpers can:
  - emit a Mermaid class diagram for kinematics classes (DHRobot, JointDH, solvers)
  - autoscan the velocity package to collect classes
  - (optionally) run pyreverse (pylint) to generate DOT/PlantUML artifacts
  - (optionally) render Graphviz images if python-graphviz is available

We keep assertions tolerant so the suite passes without optional dependencies.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

# System under test
from velocity import core, design


def test_mermaid_from_classes_core(tmp_path: Path) -> None:
    """Emit a small Mermaid diagram and verify key class/method names appear."""
    out = tmp_path / "classes.mmd"

    mermaid = design.mermaid_from_classes(
        classes=[core.JointDH, core.DHRobot, core.solvers],
        relations=[("DHRobot", "JointDH", "contains *")],
        outfile=out,
        title="Velocity – Core Classes",
    )

    text = out.read_text(encoding="utf-8")
    # Sanity: header + fenced block + our class names
    assert "Velocity – Core Classes" in text
    assert "```mermaid" in text
    assert "classDiagram" in text
    assert "class JointDH" in text
    assert "class DHRobot" in text
    # Methods important for velocity kinematics should be listed
    # (we render public functions from the class object; ensure names appear)
    assert "jacobian_geometric" in text
    assert "jacobian_analytic" in text
    assert "fk(" in text or "fk)" in text  # signature formatting can vary
    # Our explicit “contains *” relation must be emitted
    assert "DHRobot --> JointDH : contains *" in text

    # Also return value matches what was written
    assert mermaid == text


def test_autoscan_package_collects_core_classes() -> None:
    """autoscan_package should find core classes within the velocity package."""
    classes = design.autoscan_package(importlib.import_module("velocity"), include_private=False, max_depth=1)
    names = {c.__name__ for c in classes}
    # We expect at least these; more classes may appear as the project grows
    assert {"JointDH", "DHRobot", "solvers"} <= names


@pytest.mark.skipif(
    importlib.util.find_spec("pylint.pyreverse.main") is None,
    reason="pylint (pyreverse) not installed; skipping DOT/PlantUML generation test",
)
def test_run_pyreverse_generates_artifacts(tmp_path: Path) -> None:
    """If pylint is available, run pyreverse and check expected outputs exist."""
    artifacts = design.run_pyreverse(package_dir=Path(__file__).resolve().parents[1], outdir=tmp_path)

    # We accept either DOT or PlantUML (or both); check at least one classes.* file is present
    produced = set(Path(tmp_path).iterdir())
    have_dot = any(p.name == "classes.dot" for p in produced)
    have_uml = any(p.name == "classes.uml" for p in produced)
    assert have_dot or have_uml, f"Expected classes.dot or classes.uml in {tmp_path}, got {sorted(p.name for p in produced)}"


def test_default_mermaid_helper(tmp_path: Path) -> None:
    """default_mermaid should write a ready-to-embed Mermaid diagram."""
    out = tmp_path / "core_classes.mmd"
    text = design.default_mermaid(outfile=out)
    assert out.exists()
    assert text.startswith("%% Velocity – Core Classes")
    assert "class DHRobot" in text
    assert "class JointDH" in text


@pytest.mark.skipif(
    importlib.util.find_spec("graphviz") is None,
    reason="python-graphviz not installed; skipping render_graphviz test",
)
def test_graphviz_render_if_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If python-graphviz is present, ensure we can render a PNG. We use a temp outdir
    to keep the repository clean.
    """
    # Build the dot text via the same classes we tested above
    tool = _TestDiagramTool(outdir=tmp_path)
    dot_text = tool.emit_dot()  # text only
    assert "digraph" in dot_text

    # Now render a PNG (no size clamp)
    png_path = tool.render_graphviz(fmt="png", dpi=220, out_stem="classes_test")
    assert png_path.endswith(".png")
    assert Path(png_path).exists()


# --------------------------- tiny local test helper ---------------------------

class _TestDiagramTool:
    """
    Minimal façade over design.DiagramTool tuned for tests,
    keeping dependencies optional.
    """
    def __init__(self, outdir: Path) -> None:
        self._tool = design.DiagramTool(
            design.DiagramConfig(
                packages=("velocity.core", "velocity.design"),
                out_dir=outdir,
                theme="light",
                rankdir="LR",
                cluster_by_module=True,
                add_legend=False,
            )
        )

    def emit_dot(self) -> str:
        return self._tool.emit_dot()

    def render_graphviz(self, fmt: str, dpi: int, out_stem: str) -> str:
        return self._tool.render_graphviz(fmt=fmt, dpi=dpi, out_stem=out_stem)
