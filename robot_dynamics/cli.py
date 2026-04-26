
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .apis import DynamicsAPI
from .design import DHChainBuilder
from .core import State
from .io import IOMgr, IOConfig


# ------------------------------ constants ----------------------------------
DEFAULT_IN = Path("robot_dynamics/in")
DEFAULT_OUT = Path("robot_dynamics/out")


# ------------------------------ helpers ------------------------------------
def _ensure_out_path(p: Path | None, *, default_name: str) -> Path:
    """
    If *p* is None or has no parent directories, place it under DEFAULT_OUT.
    Always make parent folders.
    """
    if p is None or str(p) == "" or (not p.is_absolute() and p.parent == Path(".")):
        p = DEFAULT_OUT / default_name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_in_file(io: IOMgr, name: str | Path) -> Path:
    """
    Resolve an input file by trying robot_dynamics/in first, then robot_dynamics/out,
    and finally the path_planning as-given.
    """
    name = Path(name)
    if name.is_absolute() and name.exists():
        return name
    cand_in = io.cfg.in_dir / name
    if cand_in.exists():
        return cand_in
    cand_out = io.cfg.out_dir / name
    if cand_out.exists():
        return cand_out
    # Last resort: treat as literal path_planning (may raise later if missing)
    return name


# ------------------------------ parser -------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="robot_dynamics-dyn", description="Robot dynamics CLI (OOP + TDD)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 2R quick-run -----------------------------------------------------------
    q2r = sub.add_parser("planar2r", help="Run 2R example and save D,C,g,tau")
    q2r.add_argument("--l1", type=float, default=1.0)
    q2r.add_argument("--l2", type=float, default=1.0)
    q2r.add_argument("--m1", type=float, default=1.0)
    q2r.add_argument("--m2", type=float, default=1.0)
    q2r.add_argument("--engine", choices=["sympy", "pinocchio"], default="sympy")
    q2r.add_argument("--q", type=float, nargs=2, default=[0.2, 0.3])
    q2r.add_argument("--qd", type=float, nargs=2, default=[0.1, -0.2])
    q2r.add_argument("--qdd", type=float, nargs=2, default=[0.0, 0.0])
    q2r.add_argument("--g", type=float, default=9.81)
    # Keep back-compat default path_planning but allow bare filename; we'll normalize it
    q2r.add_argument("--out", type=Path, default=Path("robot_dynamics/out/2r_result.json"))

    # generic from YAML ------------------------------------------------------
    gy = sub.add_parser("from-yaml", help="Load model from robot_dynamics/in YAML and compute dynamics")
    gy.add_argument("name", type=str, help="YAML filename in robot_dynamics/in (bare name or path_planning)")
    gy.add_argument("--engine", choices=["sympy", "pinocchio"], default="sympy")
    gy.add_argument("--q", type=float, nargs='+', required=True)
    gy.add_argument("--qd", type=float, nargs='+', required=True)
    gy.add_argument("--qdd", type=float, nargs='+', default=None)
    gy.add_argument("--g", type=float, default=9.81)
    gy.add_argument("--out", type=Path, help="Output JSON file (default: <name>.json under robot_dynamics/out)")

    # docs skeleton ----------------------------------------------------------
    ss = sub.add_parser("sphinx-skel", help="Create a minimal Sphinx docs skeleton")
    ss.add_argument("dest", nargs="?", default="docs", help="Destination directory (default: docs)")
    ss.add_argument("--force", action="store_true", help="Overwrite existing files if present")

    return p


# ------------------------------ commands -----------------------------------
def _cmd_planar2r(args: argparse.Namespace, io: IOMgr) -> int:
    model, _ = DHChainBuilder.planar_2r(args.l1, args.l2, args.m1, args.m2)
    api = DynamicsAPI(engine=args.engine)
    state = State(q=np.array(args.q, float), qd=np.array(args.qd, float), qdd=np.array(args.qdd, float))
    res = api.run(model, state, gravity=args.g)
    out_path = _ensure_out_path(args.out, default_name="2r_result.json")
    io.save_json(out_path.name, {k: (v.tolist() if v is not None else None) for k, v in res.items()})
    print(f"Saved results to {out_path}")
    return 0


def _cmd_from_yaml(args: argparse.Namespace, io: IOMgr) -> int:
    # Resolve YAML file gracefully (in/, then out/, then as-given)
    yaml_path = _resolve_in_file(io, args.name)
    # IOMgr expects just the name typically; if a full path_planning, pass str
    model = io.model_from_yaml(str(yaml_path))
    api = DynamicsAPI(engine=args.engine)
    q = np.array(args.q, float)
    qd = np.array(args.qd, float)
    qdd = np.array(args.qdd, float) if args.qdd is not None else None
    res = api.run(model, State(q, qd, qdd), gravity=args.g)

    default_name = Path(args.name).with_suffix('.json').name
    out_path = _ensure_out_path(args.out, default_name=default_name)
    io.save_json(out_path.name, {k: (v.tolist() if v is not None else None) for k, v in res.items()})
    print(f"Saved results to {out_path}")
    return 0


def _cmd_sphinx_skel(args: argparse.Namespace) -> int:
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    conf = (
        '# Generated by robot_dynamics.cli\n'
        'project = "robot_dynamics"\n'
        'extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]\n'
        'templates_path = ["_templates"]\n'
        'exclude_patterns = []\n'
        'html_theme = "furo"\n'
    )
    index = (
        ".. robot_dynamics documentation master file\n\n"
        "Welcome to robot_dynamics's docs!\n"
        "========================\n\n"
        ".. toctree::\n"
        "   :maxdepth: 2\n"
        "   :caption: Contents:\n\n"
        "   api\n"
    )
    api = (
        "API Reference\n"
        "=============\n\n"
        ".. automodule:: robot_dynamics.core\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: robot_dynamics.design\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: robot_dynamics.apis\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: robot_dynamics.io\n   :members:\n   :undoc-members:\n\n"
    )
    makefile = (
        "# Minimal Sphinx Makefile\n"
        ".PHONY: html clean\n"
        "html:\n\t+sphinx-build -b html . _build/html\n"
        "clean:\n\t+rm -rf _build\n"
    )

    def _write(p: Path, text: str) -> None:
        if p.exists() and not args.force:
            return
        p.write_text(text, encoding="utf-8")

    (dest / "_templates").mkdir(exist_ok=True)
    (dest / "_static").mkdir(exist_ok=True)
    _write(dest / "conf.py", conf)
    _write(dest / "index.rst", index)
    _write(dest / "api.rst", api)
    _write(dest / "Makefile", makefile)
    print(str(dest))
    return 0


# ------------------------------ main ---------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    io = IOMgr(IOConfig(DEFAULT_IN, DEFAULT_OUT))

    if args.cmd == "planar2r":
        return _cmd_planar2r(args, io)
    if args.cmd == "from-yaml":
        return _cmd_from_yaml(args, io)
    if args.cmd == "sphinx-skel":
        return _cmd_sphinx_skel(args)

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
