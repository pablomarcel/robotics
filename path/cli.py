
# path/cli.py
from __future__ import annotations
"""
Path planning CLI (12.1–12.301)

Upgrades (backward-compatible):
- Keeps the original subcommands: cubic, quintic, septic, lspb, ik-2r, rot.
- Resilient I/O: resolves files relative to path/in and writes to path/out unless
  an explicit path is given. Ensures output dirs exist.
- Helpful errors and consistent JSON/CSV payloads.
- New: `sphinx-skel` subcommand to scaffold minimal Sphinx docs (like other packages).

Usage examples
--------------
python -m path.cli cubic --t0 0 --tf 2 --q0 0 --qf 1 --samples 100 --out cubic.csv
python -m path.cli lspb --t0 0 --tf 3 --q0 0 --qf 2 --vmax 1.2 --samples 120
python -m path.cli ik-2r --spec demo.json --out ik2r.csv
python -m path.cli rot --R0 R0.json --Rf Rf.json --samples 50 --out rot.json
python -m path.cli sphinx-skel docs
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .app import PathPlannerApp
from .core import BoundaryConditions

# Default I/O roots
IN_DIR = Path("path/in")
OUT_DIR = Path("path/out")


# --------------------------- helpers ---------------------------

def _ensure_out_path(p: Optional[str]) -> Path:
    """Resolve output path under OUT_DIR when a bare filename is given."""
    if not p:
        raise SystemExit("Internal error: output path is empty")
    out = Path(p)
    if not out.suffix:  # if directory or stem only, write under OUT_DIR with given name
        out = OUT_DIR / str(out)
    if not out.is_absolute():
        # If it's just a filename (no dirs), place under OUT_DIR
        if out.parent == Path('.'):
            out = OUT_DIR / out.name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _read_json_from_io(app: PathPlannerApp, name: str) -> dict:
    """Read JSON using app.io, trying IN_DIR first then OUT_DIR if relative."""
    p = Path(name)
    if p.is_absolute():
        return app.io.read_json(str(p))
    # Try IN then OUT via app.io
    # app.io.read_json understands relative filenames placed in standard locations
    try:
        return app.io.read_json(str(IN_DIR / p.name))
    except Exception:
        return app.io.read_json(str(OUT_DIR / p.name))


def _add_poly_args(sp: argparse.ArgumentParser, default_name: str) -> None:
    sp.add_argument("--t0", type=float, required=True)
    sp.add_argument("--tf", type=float, required=True)
    sp.add_argument("--q0", type=float, required=True)
    sp.add_argument("--qf", type=float, required=True)
    sp.add_argument("--qd0", type=float, default=0.0)
    sp.add_argument("--qdf", type=float, default=0.0)
    sp.add_argument("--qdd0", type=float, default=0.0)
    sp.add_argument("--qddf", type=float, default=0.0)
    sp.add_argument("--samples", type=int, default=200)
    sp.add_argument("--out", type=str, default=f"{default_name}.csv")


# --------------------------- parser ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("path-cli", description="Path planning CLI (12.1–12.301)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # cubic/quintic/septic
    for name in ("cubic", "quintic", "septic"):
        sp = sub.add_parser(name, help=f"{name} polynomial with textbook BCs")
        _add_poly_args(sp, name)

    # lspb
    sp = sub.add_parser("lspb", help="LSPB trapezoidal/triangular time-law")
    sp.add_argument("--t0", "--t-start", dest="t0", type=float, required=True)
    sp.add_argument("--tf", "--t-end", dest="tf", type=float, required=True)
    sp.add_argument("--q0", type=float, required=True)
    sp.add_argument("--qf", type=float, required=True)
    sp.add_argument("--vmax", type=float)
    sp.add_argument("--amax", type=float)
    sp.add_argument("--samples", type=int, default=200)
    sp.add_argument("--out", type=str, default="lspb.csv")

    # 2R IK along line/circle via JSON spec
    sp = sub.add_parser("ik-2r", help="Follow Cartesian path with 2R IK (spec JSON under path/in)")
    sp.add_argument("--spec", type=str, required=True, help="JSON in path/in (see docs)")
    sp.add_argument("--out", type=str, default="ik2r.csv")

    # rotation_kinematics angle-axis
    sp = sub.add_parser("rot", help="Angle-axis rotation_kinematics path between two rotation_kinematics matrices (JSON with key 'R')")
    sp.add_argument("--R0", type=str, required=True, help="JSON filename with 3x3 R0 (under path/in unless absolute)")
    sp.add_argument("--Rf", type=str, required=True, help="JSON filename with 3x3 Rf (under path/in unless absolute)")
    sp.add_argument("--samples", type=int, default=50)
    sp.add_argument("--out", type=str, default="rot.json")

    # docs skeleton
    sp = sub.add_parser("sphinx-skel", help="Create a minimal Sphinx docs skeleton (like other packages)")
    sp.add_argument("dest", nargs="?", default="docs", help="Destination directory (default: docs)")
    sp.add_argument("--force", action="store_true", help="Overwrite files if they already exist")

    return p


# --------------------------- commands --------------------------

def _do_poly(app: PathPlannerApp, which: str, args: argparse.Namespace) -> None:
    bc = BoundaryConditions(args.t0, args.tf, args.q0, args.qf, args.qd0, args.qdf, args.qdd0, args.qddf)
    if which == "lspb":
        traj = app.lspb(bc, vmax=args.vmax, amax=args.amax)
    else:
        traj = getattr(app, which)(bc)
    t = np.linspace(args.t0, args.tf, args.samples)
    samp = app.sample_1d(traj, t)
    out_path = _ensure_out_path(args.out)
    app.io.write_csv(str(out_path), t=samp.t, q=samp.q, qd=samp.qd, qdd=samp.qdd)
    print(str(out_path))


def _do_ik_2r(app: PathPlannerApp, args: argparse.Namespace) -> None:
    spec = _read_json_from_io(app, args.spec)
    try:
        l1 = float(spec["l1"]); l2 = float(spec["l2"]); elbow = spec.get("elbow", "up")
        t0 = float(spec["t0"]); tf = float(spec["tf"]); samples = int(spec.get("samples", 200))
        path = spec["path"]
    except Exception as e:
        raise SystemExit(f"Invalid spec JSON: {type(e).__name__}: {e}")
    arm = app.planar2r(l1, l2, elbow)
    t = np.linspace(t0, tf, samples)
    if path.get("type") == "line":
        X = np.linspace(path["x0"], path["x1"], t.size)
        Y = np.linspace(path["y0"], path["y1"], t.size)
    elif path.get("type") == "circle":
        cx, cy, R = path["cx"], path["cy"], path["R"]
        s = np.linspace(path.get("s0", 0.0), path.get("s1", 2*np.pi), t.size)
        X = cx + R * np.cos(s); Y = cy + R * np.sin(s)
    else:
        raise SystemExit("spec['path']['type'] must be 'line' or 'circle'")
    th1, th2 = arm.ik(X, Y)
    out_path = _ensure_out_path(args.out)
    app.io.write_csv(str(out_path), t=t, X=X, Y=Y, th1=th1, th2=th2)
    print(str(out_path))


def _do_rot(app: PathPlannerApp, args: argparse.Namespace) -> None:
    R0j = _read_json_from_io(app, args.R0)
    Rfj = _read_json_from_io(app, args.Rf)
    try:
        R0 = np.array(R0j["R"], dtype=float).reshape(3, 3)
        Rf = np.array(Rfj["R"], dtype=float).reshape(3, 3)
    except Exception as e:
        raise SystemExit(f"R0/Rf JSON must contain key 'R' with a 3x3 array: {type(e).__name__}: {e}")
    path = app.angle_axis_path(R0, Rf)
    s = np.linspace(0.0, 1.0, args.samples)
    Rseq = path.R(s).tolist()
    out_path = _ensure_out_path(args.out)
    app.io.write_json(str(out_path), {"R": Rseq})
    print(str(out_path))


def _do_sphinx_skel(dest: str, force: bool = False) -> None:
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    conf = (
        '# Generated by path.cli\n'
        'project = "path"\n'
        'extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]\n'
        'templates_path = ["_templates"]\n'
        'exclude_patterns = []\n'
        'html_theme = "furo"\n'
    ).replace('\\n', '\n')
    index = (
        ".. path documentation master file\n\n"
        "Welcome to path's docs!\n"
        "=======================\n\n"
        ".. toctree::\n"
        "   :maxdepth: 2\n"
        "   :caption: Contents:\n\n"
        "   api\n"
    ).replace('\\n', '\n')
    api = (
        "API Reference\n"
        "=============\n\n"
        ".. automodule:: path.app\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: path.core\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: path.io\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: path.utils\n   :members:\n   :undoc-members:\n\n"
    ).replace('\\n', '\n')
    makefile = (
        "# Minimal Sphinx Makefile\n"
        ".PHONY: html clean\n"
        "html:\n\t+sphinx-build -b html . _build/html\n"
        "clean:\n\t+rm -rf _build\n"
    ).replace('\\n', '\n')

    def _write(path: Path, text: str) -> None:
        if path.exists() and not force:
            return
        path.write_text(text, encoding="utf-8")

    (d / "_templates").mkdir(exist_ok=True)
    (d / "_static").mkdir(exist_ok=True)
    _write(d / "conf.py", conf)
    _write(d / "index.rst", index)
    _write(d / "api.rst", api)
    _write(d / "Makefile", makefile)
    print(str(d))


# ----------------------------- main ----------------------------

def main(argv=None):
    argv = argv or sys.argv[1:]
    args = build_parser().parse_args(argv)
    # docs scaffold is standalone (no app dependency)
    if args.cmd == "sphinx-skel":
        _do_sphinx_skel(args.dest, force=getattr(args, "force", False))
        return

    app = PathPlannerApp()

    if args.cmd in ("cubic", "quintic", "septic"):
        _do_poly(app, args.cmd, args)
    elif args.cmd == "lspb":
        _do_poly(app, "lspb", args)
    elif args.cmd == "ik-2r":
        _do_ik_2r(app, args)
    elif args.cmd == "rot":
        _do_rot(app, args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
