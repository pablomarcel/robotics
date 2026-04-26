# applied_dynamics/cli.py
"""
Command-line entry points for the **applied_dynamics** dynamics module.

Subcommands
-----------
design      - list or instantiate prebuilt dynamics presets
diagram     - generate class diagrams (DOT/PlantUML/JSON/Graphviz)
sphinx-skel - create/refresh Sphinx docs skeleton in applied_dynamics/docs

# Simple, test-oriented shortcuts (numeric presets):
pendulum | spherical | planar2r | absorber
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import argparse
import json
import sys

from .design import DesignLibrary, build_pendulum, build_spherical_pendulum, build_planar_2r, build_cart_absorber
from .tools.diagram import (
    DiagramConfig,
    DiagramTool,
    render_dot as _render_dot,          # back-compat if someone imports cli.render_dot
    render_puml as _render_puml,
)

__all__ = ["main"]


# ------------------------------ helpers ------------------------------------ #

def _print_model_summary(model) -> None:
    """Pretty minimal, but testable, one-screen summary."""
    cls = type(model).__name__
    fields = {}
    if hasattr(model, "__dict__"):
        fields = {k: v for k, v in model.__dict__.items() if not k.startswith("_")}
    print(f"Model: {cls}")
    if fields:
        print("Parameters:")
        for k, v in fields.items():
            print(f"  - {k}: {v}")
    if hasattr(model, "q"):
        print(f"Generalized coordinates: {getattr(model, 'q')}")
    if hasattr(model, "qd"):
        print(f"Generalized speeds: {getattr(model, 'qd')}")


def _parse_packages_csv(s: str):
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _make_tool(args) -> DiagramTool:
    cfg = DiagramConfig(
        packages=_parse_packages_csv(args.packages),
        out_dir=Path(args.outdir),
        theme=args.theme,
        rankdir=args.rankdir,
        cluster_by_module=not args.no_cluster,
        add_legend=args.legend,
    )
    return DiagramTool(cfg)


def _write(path: Path, text: str, *, force: bool) -> None:
    """Write helper that respects --force and ensures parent dirs."""
    if not path.exists() or force:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


# ------------------------------ CLI wiring --------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="applied_dynamics.cli", description="Applied dynamics CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # -- design --------------------------------------------------------------
    d = sub.add_parser("design", help="Preset dynamics builders")
    d.add_argument("--list", action="store_true", help="List available presets")
    d.add_argument("--preset", help="Preset name (e.g., planar2r_num)")
    d.add_argument("--export", help="If given, write a small JSON summary to this path")

    # -- diagram -------------------------------------------------------------
    g = sub.add_parser("diagram", help="Generate class diagrams for the 'applied_dynamics' package")
    subg = g.add_subparsers(dest="gcmd", required=True)

    def add_common(a):
        a.add_argument("--packages", default="applied_dynamics.core,applied_dynamics.dynamics,applied_dynamics.models,applied_dynamics.io,applied_dynamics.utils,applied_dynamics.app,applied_dynamics.apis,applied_dynamics.tools")
        a.add_argument("--outdir", default="applied_dynamics/out")
        a.add_argument("--theme", default="light", choices=["light", "dark"])
        a.add_argument("--rankdir", default="LR", choices=["LR", "TB"])
        a.add_argument("--legend", action="store_true")
        a.add_argument("--no-cluster", action="store_true")

    dt = subg.add_parser("dot", help="Emit Graphviz DOT text")
    add_common(dt)
    dt.add_argument("--out", default="classes.dot")

    pu = subg.add_parser("plantuml", help="Emit PlantUML text")
    add_common(pu)
    pu.add_argument("--out", default="classes.puml")

    jj = subg.add_parser("json", help="Export discovered model JSON")
    add_common(jj)
    jj.add_argument("--out", default="classes.json")

    gv = subg.add_parser("graphviz", help="Render a diagram via python-graphviz")
    add_common(gv)
    gv.add_argument("--fmt", default="png", choices=["png", "svg", "pdf"])
    gv.add_argument("--dpi", type=int, default=220)
    gv.add_argument("--outstem", default="classes")

    aa = subg.add_parser("all", help="Emit JSON, DOT, PlantUML and try Graphviz")
    add_common(aa)

    # -- sphinx docs skeleton ------------------------------------------------
    sk = sub.add_parser("sphinx-skel", help="Create/refresh Sphinx docs skeleton in applied_dynamics/docs")
    sk.add_argument("dest", nargs="?", default="applied_dynamics/docs", help="Destination docs dir (default: applied_dynamics/docs)")
    sk.add_argument("--force", action="store_true", help="Overwrite existing files if present")

    # -- simple numeric shortcuts (used by smoke tests) ---------------------
    sub.add_parser("pendulum", help="Build simple pendulum (numeric defaults) and print a summary")
    sub.add_parser("spherical", help="Build spherical pendulum (numeric defaults) and print a summary")
    sub.add_parser("planar2r", help="Build planar 2R (numeric defaults) and print a summary")
    sub.add_parser("absorber", help="Build cart–pendulum absorber (numeric defaults) and print a summary")

    return p


# ------------------------------ main --------------------------------------- #

def main(argv: Optional[Iterable[str]] = None) -> int:  # return int for test expectations
    parser = _build_parser()
    args = parser.parse_args(argv)

    # design subcommand
    if args.cmd == "design":
        lib = DesignLibrary()
        if args.list and not args.preset:
            for name in lib.available():
                print(name)
            return 0
        if not args.preset:
            parser.error("design requires --list or --preset NAME")
        model = lib.create(args.preset)
        _print_model_summary(model)
        if args.export:
            payload = {
                "preset": args.preset,
                "type": type(model).__name__,
                "fields": {k: str(v) for k, v in getattr(model, "__dict__", {}).items() if not k.startswith("_")},
            }
            path = Path(args.export)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(str(path))
        return 0

    # diagram subcommand
    if args.cmd == "diagram":
        tool = _make_tool(args)

        if args.gcmd == "dot":
            print(tool.emit_dot(out_file=args.out))
            return 0
        if args.gcmd == "plantuml":
            print(tool.emit_plantuml(out_file=args.out))
            return 0
        if args.gcmd == "json":
            print(tool.export_model_json(out_file=args.out))
            return 0
        if args.gcmd == "graphviz":
            try:
                print(tool.render_graphviz(fmt=args.fmt, dpi=args.dpi, out_stem=args.outstem))
            except RuntimeError as e:
                raise SystemExit(str(e))
            return 0
        if args.gcmd == "all":
            arts = tool.render_all()
            for k, v in arts.items():
                print(f"{k}: {v}")
            return 0

    # sphinx docs skeleton ---------------------------------------------------
    if args.cmd == "sphinx-skel":
        dest = Path(args.dest)
        dest.mkdir(parents=True, exist_ok=True)

        conf = (
            '# Generated by applied_dynamics.cli\n'
            'project = "applied_dynamics"\n'
            'extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]\n'
            'templates_path = ["_templates"]\n'
            'exclude_patterns = []\n'
            'html_theme = "furo"\n'
        )
        index = (
            ".. applied_dynamics documentation master file\n\n"
            "Welcome to applied_dynamics's docs!\n"
            "==========================\n\n"
            ".. toctree::\n"
            "   :maxdepth: 2\n"
            "   :caption: Contents:\n\n"
            "   api\n"
        )
        api_rst = (
            "API Reference\n"
            "=============\n\n"
            ".. automodule:: applied_dynamics.app\n   :members:\n   :undoc-members:\n\n"
            ".. automodule:: applied_dynamics.core\n   :members:\n   :undoc-members:\n\n"
            ".. automodule:: applied_dynamics.dynamics\n   :members:\n   :undoc-members:\n\n"
            ".. automodule:: applied_dynamics.models\n   :members:\n   :undoc-members:\n\n"
            ".. automodule:: applied_dynamics.io\n   :members:\n   :undoc-members:\n\n"
            ".. automodule:: applied_dynamics.utils\n   :members:\n   :undoc-members:\n\n"
            ".. automodule:: applied_dynamics.tools.diagram\n   :members:\n   :undoc-members:\n\n"
            ".. automodule:: applied_dynamics.apis\n   :members:\n   :undoc-members:\n"
        )
        makefile = (
            "# Minimal Sphinx Makefile\n"
            ".PHONY: html clean\n"
            "html:\n\t+sphinx-build -b html . _build/html\n"
            "clean:\n\t+rm -rf _build\n"
        )

        _write(dest / "conf.py", conf, force=args.force)
        _write(dest / "index.rst", index, force=args.force)
        _write(dest / "api.rst", api_rst, force=args.force)
        _write(dest / "Makefile", makefile, force=args.force)
        (dest / "_templates").mkdir(exist_ok=True)
        (dest / "_static").mkdir(exist_ok=True)
        print(str(dest))
        return 0

    # simple numeric shortcuts ------------------------------------------------
    g = 9.81
    if args.cmd == "pendulum":
        model = build_pendulum(m=1.0, l=1.0, g=g)
        _print_model_summary(model)
        return 0
    if args.cmd == "spherical":
        model = build_spherical_pendulum(m=1.0, l=1.0, g=g)
        _print_model_summary(model)
        return 0
    if args.cmd == "planar2r":
        model = build_planar_2r(m1=1.0, m2=1.0, l1=1.0, l2=0.8, g=g)
        _print_model_summary(model)
        return 0
    if args.cmd == "absorber":
        model = build_cart_absorber(M=5.0, m=0.5, l=0.6, k=50.0, g=g)
        _print_model_summary(model)
        return 0

    # Shouldn't reach here (argparse enforces subcommands)
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
