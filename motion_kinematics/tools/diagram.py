# motion_kinematics/tools/diagram.py
"""
Tools for generating class diagrams of the `motion_kinematics` package.

This module provides:
- DiagramTool: an OOP facade to discover classes and emit diagrams
- A small CLI (python -m motion_kinematics.tools.diagram ...) for scripts/CI

Backends supported (all optional except discover/JSON):
- Graphviz (PNG/SVG/PDF)         -> requires the 'graphviz' python package
- pyreverse (classes/packages)   -> requires 'pylint' + 'pyreverse' on PATH
- PlantUML (.puml text)          -> plain text; render externally if desired
- Mermaid (.mmd text)            -> plain text; render externally if desired
- JSON model (.json)             -> great for snapshot tests

Typical usage
-------------
from motion_kinematics.tools.diagram import DiagramTool, DiagramConfig

tool = DiagramTool()  # defaults to the 'motion_kinematics' package and motion_kinematics/out
model = tool.discover()
paths = tool.render_all()  # produce a sensible set of outputs

# CLI:
#   python -m motion_kinematics.tools.diagram discover --package motion_kinematics
#   python -m motion_kinematics.tools.diagram graphviz  --fmt png --out motion_kinematics/out/classes
#   python -m motion_kinematics.tools.diagram pyreverse --out motion_kinematics/out
#   python -m motion_kinematics.tools.diagram plantuml  --out motion_kinematics/out/classes.puml
#   python -m motion_kinematics.tools.diagram mermaid   --out motion_kinematics/out/classes.mmd
#   python -m motion_kinematics.tools.diagram json      --out motion_kinematics/out/classes.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json
import os

from motion_kinematics.utils import timing
from motion_kinematics.design import (
    discover as _discover,
    render_graphviz as _render_graphviz,
    generate_pyreverse as _generate_pyreverse,
    emit_plantuml as _emit_plantuml,
    emit_mermaid as _emit_mermaid,
    save_model_json as _save_model_json,
    Model,
)


# ------------------------------- configuration --------------------------------

@dataclass(frozen=True)
class DiagramConfig:
    """
    Configuration for diagram generation.

    Attributes
    ----------
    package : str
        Top-level package to scan (defaults to 'motion_kinematics').
    out_dir : Path
        Base output directory for generated assets.
    """
    package: str = "motion_kinematics"
    out_dir: Path = Path("motion_kinematics/out")


# ---------------------------------- service -----------------------------------

class DiagramTool:
    """
    Facade service for class diagram generation.

    Notes
    -----
    - All methods are pure w.r.t. discovery (no global state) and return paths
      for easy assertions in unit tests.
    - Graphviz/pyreverse are optional; methods raise RuntimeError with a clear
      message if missing, which you can mark xfail/skip in tests.
    """

    def __init__(self, config: Optional[DiagramConfig] = None) -> None:
        self.cfg = config or DiagramConfig()
        self._ensure_dir(self.cfg.out_dir)

    # ----------------------------- helpers --------------------------------
    @staticmethod
    def _ensure_dir(path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    # ----------------------------- actions --------------------------------
    @timing
    def discover(self) -> Model:
        """
        Discover classes under `cfg.package`.
        """
        return _discover(self.cfg.package)

    @timing
    def render_graphviz(self, *, fmt: str = "png", rankdir: str = "LR",
                        out_stem: str = "classes") -> str:
        """
        Render a Graphviz diagram. Returns the produced file path.

        Parameters
        ----------
        fmt : {'png','svg','pdf',...}
        rankdir : {'LR','TB'}
        out_stem : str
            Output path (stem only) within out_dir; extension is added by Graphviz.
        """
        model = self.discover()
        out_path = str((self.cfg.out_dir / out_stem).as_posix())
        return _render_graphviz(model, out_path=out_path, fmt=fmt, rankdir=rankdir)

    @timing
    def render_pyreverse(self, *, out_dir: Optional[Path] = None, fmt: str = "png") -> List[str]:
        """
        Run pyreverse and move outputs into `out_dir` (defaults to cfg.out_dir).
        Returns list of produced files.
        """
        target = out_dir or self.cfg.out_dir
        self._ensure_dir(target)
        return _generate_pyreverse(self.cfg.package, out_dir=str(target), fmt=fmt)

    @timing
    def emit_plantuml(self, *, out_file: str = "classes.puml") -> str:
        """
        Emit PlantUML text file and return its path.
        """
        model = self.discover()
        path = str((self.cfg.out_dir / out_file).as_posix())
        return _emit_plantuml(model, path)

    @timing
    def emit_mermaid(self, *, out_file: str = "classes.mmd") -> str:
        """
        Emit Mermaid text file and return its path.
        """
        model = self.discover()
        path = str((self.cfg.out_dir / out_file).as_posix())
        return _emit_mermaid(model, path)

    @timing
    def export_model_json(self, *, out_file: str = "classes.json") -> str:
        """
        Save the discovered model as JSON for snapshot tests; returns path.
        """
        model = self.discover()
        path = str((self.cfg.out_dir / out_file).as_posix())
        return _save_model_json(model, path)

    @timing
    def render_all(self) -> Dict[str, List[str] | str]:
        """
        Produce a reasonable set of outputs. Returns a dict of artifact paths.

        This tries Graphviz (PNG) and PlantUML/Mermaid/JSON.
        Pyreverse is optional and will be attempted if available.
        """
        artifacts: Dict[str, List[str] | str] = {}
        # JSON model (always available)
        artifacts["json"] = self.export_model_json()

        # PlantUML/Mermaid (text-only)
        artifacts["plantuml"] = self.emit_plantuml()
        artifacts["mermaid"] = self.emit_mermaid()

        # Graphviz (optional)
        try:
            artifacts["graphviz"] = self.render_graphviz(fmt="png", rankdir="LR")
        except RuntimeError:
            artifacts["graphviz"] = "unavailable (graphviz not installed)"

        # pyreverse (optional)
        try:
            artifacts["pyreverse"] = self.render_pyreverse(fmt="png")
        except RuntimeError:
            artifacts["pyreverse"] = ["unavailable (pyreverse not on PATH)"]

        return artifacts


# ------------------------------------ CLI -------------------------------------

def _build_cli():
    import argparse

    p = argparse.ArgumentParser(prog="motion_kinematics.tools.diagram", description="Class diagram generator for the motion_kinematics package")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared
    def add_shared(a):
        a.add_argument("--package", default="motion_kinematics", help="Top-level package to inspect (default: motion_kinematics)")
        a.add_argument("--outdir", default="motion_kinematics/out", help="Output directory (default: motion_kinematics/out)")

    # discover
    d = sub.add_parser("discover", help="Discover classes and print JSON to stdout")
    add_shared(d)

    # graphviz
    g = sub.add_parser("graphviz", help="Render a Graphviz diagram")
    add_shared(g)
    g.add_argument("--fmt", default="png", choices=["png", "svg", "pdf"], help="Output format")
    g.add_argument("--rankdir", default="LR", choices=["LR", "TB"], help="Layout direction")
    g.add_argument("--stem", default="classes", help="Output stem (extension added automatically)")

    # pyreverse
    r = sub.add_parser("pyreverse", help="Run pyreverse and place outputs in outdir")
    add_shared(r)
    r.add_argument("--fmt", default="png", help="Output format (png by default)")

    # plantuml
    u = sub.add_parser("plantuml", help="Emit PlantUML file")
    add_shared(u)
    u.add_argument("--out", default="classes.puml", help="Output filename (within outdir)")

    # mermaid
    m = sub.add_parser("mermaid", help="Emit Mermaid file")
    add_shared(m)
    m.add_argument("--out", default="classes.mmd", help="Output filename (within outdir)")

    # json
    j = sub.add_parser("json", help="Save JSON model file")
    add_shared(j)
    j.add_argument("--out", default="classes.json", help="Output filename (within outdir)")

    # all
    a = sub.add_parser("all", help="Render JSON, PlantUML, Mermaid, Graphviz (if available), pyreverse (if available)")
    add_shared(a)

    return p


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)

    cfg = DiagramConfig(package=args.package, out_dir=Path(args.outdir))
    tool = DiagramTool(cfg)

    if args.cmd == "discover":
        model = tool.discover()
        data = {
            "package": model.package,
            "classes": [
                {"qualname": c.qualname, "modname": c.modname, "name": c.name, "bases": list(c.bases), "doc": c.doc}
                for c in model.classes
            ],
        }
        print(json.dumps(data, indent=2))
        return

    if args.cmd == "graphviz":
        path = tool.render_graphviz(fmt=args.fmt, rankdir=args.rankdir, out_stem=args.stem)
        print(path)
        return

    if args.cmd == "pyreverse":
        paths = tool.render_pyreverse(fmt=args.fmt)
        print("\n".join(paths))
        return

    if args.cmd == "plantuml":
        print(tool.emit_plantuml(out_file=args.out))
        return

    if args.cmd == "mermaid":
        print(tool.emit_mermaid(out_file=args.out))
        return

    if args.cmd == "json":
        print(tool.export_model_json(out_file=args.out))
        return

    if args.cmd == "all":
        artifacts = tool.render_all()
        # Print a compact summary for CI logs
        for k, v in artifacts.items():
            print(f"{k}: {v}")
        return


if __name__ == "__main__":  # pragma: no cover
    main()
