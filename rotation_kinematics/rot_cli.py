#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for :mod:`rotation_kinematics.rot_cli`.

The CLI supports two usage styles:

1. Modern subcommand form::

       python -m rotation_kinematics.rot_cli compose global zyx "30,20,10" --degrees
       python -m rotation_kinematics.rot_cli sphinx-skel rotation_kinematics/docs

2. Historical no-subcommand form for the common compose workflow::

       python -m rotation_kinematics.rot_cli global zyx "30,20,10" --degrees

The ``sphinx-skel`` helper generates a conservative, GitHub Pages friendly
Sphinx documentation skeleton with dynamic RST heading underline lengths,
tracked support directories, and an API page that includes only modules that
are actually importable in the current Python environment.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------- Import shim so `python rot_cli.py` works with absolute imports ----------
if __package__ in (None, ""):
    # Running as a script: add repository root to sys.path and import absolute modules.
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    from rotation_kinematics.rot_utils import ensure_dir, parse_floats
    from rotation_kinematics.rot_design import HELP_SEQ
    from rotation_kinematics import rot_core as core
    from rotation_kinematics import rot_io as rio
    from rotation_kinematics import rot_closedform as cf
else:
    from .rot_utils import ensure_dir, parse_floats
    from .rot_design import HELP_SEQ
    from . import rot_core as core
    from . import rot_io as rio
    from . import rot_closedform as cf


# -----------------------------------------------------------------------------
# Project/package metadata for docs generation
# -----------------------------------------------------------------------------
_PACKAGE = "rotation_kinematics"
_PROJECT = "Robotics - rotation_kinematics"
_AUTHOR = "Robotics"

_REQUIRED_MODULES = [
    "rotation_kinematics.rot_cli",
    "rotation_kinematics.rot_core",
]

_OPTIONAL_MODULES = [
    "rotation_kinematics.rot_design",
    "rotation_kinematics.rot_io",
    "rotation_kinematics.rot_utils",
    "rotation_kinematics.rot_closedform",
    "rotation_kinematics.arch_ascii",
]

_AUTODOC_MOCK_IMPORTS = [
    "matplotlib",
    "matplotlib.pyplot",
    "numpy",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.transform",
    "sympy",
    "yaml",
]


# -----------------------------------------------------------------------------
# Defaults and I/O conveniences
# -----------------------------------------------------------------------------
OUT_DIR = os.path.join("rotation_kinematics", "out")
IN_DIR = os.path.join("rotation_kinematics", "in")
ensure_dir(OUT_DIR)
ensure_dir(IN_DIR)


def _resolve_in(path_like: str) -> str:
    """
    Resolve an input file by searching ``rotation_kinematics/in``, then
    ``rotation_kinematics/out``, then using the provided path as-is.
    """
    cands = [
        os.path.join(IN_DIR, path_like),
        os.path.join(OUT_DIR, path_like),
        path_like,
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[-1]


def _resolve_out(path_like: str | None, default_name: str) -> str:
    """
    Resolve an output path. If ``path_like`` is ``None`` or a bare filename,
    write under ``rotation_kinematics/out``. Parents are created as needed.
    """
    if not path_like:
        out = os.path.join(OUT_DIR, default_name)
    elif os.path.dirname(path_like) == "":
        out = os.path.join(OUT_DIR, path_like)
    else:
        out = path_like
    ensure_dir(os.path.dirname(out))
    return out


# -----------------------------------------------------------------------------
# Sphinx skeleton helpers
# -----------------------------------------------------------------------------
def _module_is_importable(module_name: str) -> bool:
    """Return ``True`` only when a module can actually be imported."""
    try:
        if importlib.util.find_spec(module_name) is None:
            return False
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def _available_modules() -> list[str]:
    """Return package modules that are safe for Sphinx autodoc to import."""
    modules: list[str] = []
    for mod in [*_REQUIRED_MODULES, *_OPTIONAL_MODULES]:
        if _module_is_importable(mod):
            modules.append(mod)
    return modules


def _rst_heading(text: str, underline: str = "=") -> str:
    """Return a reStructuredText heading with matching underline length."""
    return f"{text}\n{underline * len(text)}\n\n"


def _write_if_needed(path: Path, text: str, *, force: bool = False) -> bool:
    """Write ``text`` unless ``path`` exists and overwrite is disabled."""
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _ensure_sphinx_support_dirs(dest: Path) -> None:
    """Create Sphinx support directories and tracked placeholder files."""
    for dirname in ("_templates", "_static"):
        folder = dest / dirname
        folder.mkdir(parents=True, exist_ok=True)
        gitkeep = folder / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")


def _build_conf_py() -> str:
    """Build a conservative Sphinx ``conf.py`` for GitHub Pages deployments."""
    return f'''# Generated by {_PACKAGE}.rot_cli
from __future__ import annotations

import sys
from pathlib import Path

# docs -> rotation_kinematics -> repository root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = {_PROJECT!r}
author = {_AUTHOR!r}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_static_path = ["_static"]

autodoc_typehints = "description"
autodoc_mock_imports = {_AUTODOC_MOCK_IMPORTS!r}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
'''


def _build_index_rst() -> str:
    """Build a Sphinx ``index.rst`` page with safe heading underline lengths."""
    title = f"Welcome to {_PACKAGE}'s documentation"
    return (
        f".. {_PACKAGE} documentation master file\n\n"
        + _rst_heading(title, "=")
        + ".. toctree::\n"
        + "   :maxdepth: 2\n"
        + "   :caption: Contents:\n\n"
        + "   api\n"
    )


def _build_api_rst() -> str:
    """Build an API page that includes only importable modules."""
    parts: list[str] = [_rst_heading("API Reference", "=")]
    modules = _available_modules()

    if not modules:
        parts.append(
            "No modules were importable when this API page was generated.\n\n"
            "Regenerate the Sphinx skeleton from an environment where the package "
            "can be imported, or install the package dependencies before building docs.\n"
        )
        return "".join(parts)

    for mod in modules:
        parts.append(_rst_heading(mod, "-"))
        parts.append(f".. automodule:: {mod}\n")
        parts.append("   :members:\n")
        parts.append("   :undoc-members:\n")
        parts.append("   :show-inheritance:\n\n")

    return "".join(parts)


def _build_makefile() -> str:
    """Build the minimal project-standard Sphinx Makefile."""
    return """# Minimal Sphinx Makefile
.PHONY: html clean
html:
	+sphinx-build -b html . _build/html
clean:
	+rm -rf _build
"""


def _write_sphinx_skeleton(dest: Path, *, force: bool = False) -> list[Path]:
    """Create or update a deploy-safe Sphinx documentation skeleton."""
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    _ensure_sphinx_support_dirs(dest)

    files = {
        dest / "conf.py": _build_conf_py(),
        dest / "index.rst": _build_index_rst(),
        dest / "api.rst": _build_api_rst(),
        dest / "Makefile": _build_makefile(),
    }

    written: list[Path] = []
    for path, text in files.items():
        if _write_if_needed(path, text, force=force):
            written.append(path)
    return written


# -----------------------------------------------------------------------------
# Small parse/print helpers
# -----------------------------------------------------------------------------
def _angles_arg(s: str, degrees: bool) -> List[float]:
    """Parse exactly three comma-separated angle values."""
    vals = parse_floats(s)
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("need 3 comma-separated values")
    return vals


def _vec_arg(s: str) -> List[float]:
    """Parse exactly three comma-separated vector components."""
    vals = parse_floats(s)
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("need 3 comma-separated values")
    return vals


def _fmt_ndarray(arr: np.ndarray) -> str:
    """Format a NumPy array with compact fixed precision."""
    return np.array2string(arr, formatter={"float_kind": lambda x: f"{x: .6f}"})


def _print_sympy_matrix(M: Any, names: List[str], title: str) -> None:
    """Pretty-print a SymPy matrix without the ``Matrix(...)`` wrapper."""
    try:
        import sympy as sp
    except Exception:
        print(title)
        print("variables:", ", ".join(names))
        print(M)
        return

    rows = []
    for i in range(M.rows):
        row_elems = [sp.sstr(sp.simplify(M[i, j])) for j in range(M.cols)]
        rows.append("  [" + ", ".join(row_elems) + "]")
    print(title)
    print("variables:", ", ".join(names))
    print("[\n" + "\n".join(rows) + "\n]")


# -----------------------------------------------------------------------------
# Batch task execution
# -----------------------------------------------------------------------------
def _run_one_task(t: Dict[str, Any]) -> None:
    """Execute a single batch task dict."""
    cmd = t.get("cmd")

    if cmd == "compose":
        Robj = core.build_matrix(
            t["mode"], t["seq"], t["angles"], degrees=bool(t.get("degrees", False))
        )
        print(_fmt_ndarray(Robj.as_matrix()))
        if "save" in t:
            out = _resolve_out(t["save"], "R.csv")
            core.save_R(out, Robj)
            print(f"saved -> {out}")
        return

    if cmd == "transform":
        Robj = core.build_matrix(
            t["mode"], t["seq"], t["angles"], degrees=bool(t.get("degrees", False))
        )
        if "points_file" in t:
            pf = _resolve_in(t["points_file"])
            P = rio.read_points_csv(pf)
        else:
            P = np.array(t.get("points", []), dtype=float)
            if P.size == 0:
                P = np.eye(3)
        Pg = core.transform_points(Robj, P)
        print(_fmt_ndarray(Pg))
        if "save" in t:
            out = _resolve_out(t["save"], "points_out.csv")
            rio.write_points_csv(out, Pg)
            print(f"saved -> {out}")
        return

    if cmd == "passive":
        Robj = core.build_matrix(
            t["mode"], t["seq"], t["angles"], degrees=bool(t.get("degrees", False))
        )
        P = np.array(t.get("points", []), dtype=float)
        if P.size == 0:
            P = np.eye(3)
        Pb = core.passive_transform(Robj, P)
        print(_fmt_ndarray(Pb))
        return

    if cmd == "angvel":
        w = core.angvel_from_rates(
            t["seq"],
            t["angles"],
            t["rates"],
            convention=t["mode"],
            degrees=bool(t.get("degrees", False)),
            frame=t.get("frame", "body"),
        )
        print(_fmt_ndarray(w))
        return

    if cmd == "rates":
        qdot = core.rates_from_angvel(
            t["seq"],
            t["angles"],
            t["omega"],
            convention=t["mode"],
            degrees=bool(t.get("degrees", False)),
            frame=t.get("frame", "body"),
        )
        print(_fmt_ndarray(qdot))
        return

    if cmd == "repeat":
        Robj = core.build_matrix(
            t["mode"], t["seq"], t["angles"], degrees=bool(t.get("degrees", False))
        )
        Rm = core.repeat_rotation(Robj, int(t["m"]))
        print(_fmt_ndarray(Rm.as_matrix()))
        return

    if cmd == "align":
        Robj = core.align_body_x(t["u"])
        print(_fmt_ndarray(Robj.as_matrix()))
        return

    raise ValueError(f"Unknown batch cmd: {cmd}")


# -----------------------------------------------------------------------------
# Parser construction
# -----------------------------------------------------------------------------
def _add_compose_arguments(pc: argparse.ArgumentParser) -> None:
    """Add compose workflow arguments to a parser."""
    pc.add_argument("mode", choices=["global", "local"], help="Global/extrinsic or local/intrinsic rotations.")
    pc.add_argument("seq", help=HELP_SEQ)
    pc.add_argument("angles", type=str, help="Comma list of three angles.")
    pc.add_argument("--degrees", action="store_true", help="Angles are degrees; default is radians.")
    pc.add_argument(
        "--save",
        type=str,
        default=None,
        help="CSV filename or path to save the matrix. Bare names are written under rotation_kinematics/out.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(
        prog="rotation_kinematics.rot_cli",
        description=(
            "Rotation Kinematics CLI for composition, decomposition, active/passive "
            "point transforms, angular velocity/rate maps, closed-form E matrices, "
            "batch jobs, and Sphinx skeleton generation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    sp = p.add_subparsers(dest="cmd")

    pc = sp.add_parser(
        "compose",
        help="build a rotation matrix from a sequence and angles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    _add_compose_arguments(pc)

    pd = sp.add_parser(
        "decompose",
        help="extract angles from a rotation matrix for a given sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    pd.add_argument("mode", choices=["global", "local"])
    pd.add_argument("seq", help=HELP_SEQ)
    g = pd.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-angles", type=str, help="Compose first from these angles, using the same sequence/mode.")
    g.add_argument("--from-csv", type=str, help="Matrix CSV filename resolved via in -> out -> literal path.")
    pd.add_argument("--degrees", action="store_true")

    pt = sp.add_parser(
        "transform",
        help="active transform points: rG = R rB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    pt.add_argument("mode", choices=["global", "local"])
    pt.add_argument("seq", help=HELP_SEQ)
    pt.add_argument("angles", type=str)
    pt.add_argument("--degrees", action="store_true")
    pt.add_argument("--points", type=str, help="CSV of Nx3 points or inline rows 'x,y,z|...'.", default=None)
    pt.add_argument("--save", type=str, default=None, help="CSV filename/path to save transformed points.")

    ppv = sp.add_parser(
        "passive",
        help="passive coordinate change: rB = R^T rG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    ppv.add_argument("mode", choices=["global", "local"])
    ppv.add_argument("seq", help=HELP_SEQ)
    ppv.add_argument("angles", type=str)
    ppv.add_argument("--degrees", action="store_true")
    ppv.add_argument("--points", type=str, default=None, help="CSV filename or inline rows 'x,y,z|...'.")

    pr = sp.add_parser(
        "repeat",
        help="repeat/exponentiate a rotation: R^m",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    pr.add_argument("mode", choices=["global", "local"])
    pr.add_argument("seq", help=HELP_SEQ)
    pr.add_argument("angles", type=str)
    pr.add_argument("m", type=int)
    pr.add_argument("--degrees", action="store_true")

    pa = sp.add_parser(
        "align",
        help="build R that aligns body x-axis to vector u and y into the XY-plane",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    pa.add_argument("u", type=str, help="Comma list for target vector u.")

    pk = sp.add_parser(
        "check",
        help="check orthogonality and det=1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    pk.add_argument("--from-angles", type=str, help="Compose first from these angles.")
    pk.add_argument("--mode", choices=["global", "local"], default="global")
    pk.add_argument("--seq", default="zyx")
    pk.add_argument("--degrees", action="store_true")
    pk.add_argument("--from-csv", type=str, help="Matrix CSV filename resolved via in -> out -> literal path.")

    pe = sp.add_parser(
        "E",
        help="closed-form E(q) such that omega = E(q) qdot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    pe.add_argument("seq", choices=["zyx", "zxz", "zyz"], help="RPY ZYX or proper Euler ZXZ/ZYZ.")
    pe.add_argument("--convention", choices=["local", "global"], default="local", help="Angle convention.")
    pe.add_argument("--frame", choices=["body", "space"], default="body", help="Frame of omega.")
    pe.add_argument(
        "--rpy-order",
        action="store_true",
        help="For zyx only, also print RPY order [phi, theta, psi] columns.",
    )

    pb = sp.add_parser(
        "batch",
        help="run a JSON/YAML job file with multiple tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    pb.add_argument("file", type=str, help="Filename resolved via rotation_kinematics/in -> out -> literal path.")

    pv = sp.add_parser(
        "angvel",
        help="compute omega from angle rates for any sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    pv.add_argument("mode", choices=["global", "local"])
    pv.add_argument("seq", help=HELP_SEQ)
    pv.add_argument("angles", type=str)
    pv.add_argument("rates", type=str, help="Comma list of angle rates.")
    pv.add_argument("--frame", choices=["body", "space"], default="body")
    pv.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret both angles and rates in degrees/deg/s; default is radians/rad/s.",
    )

    pvr = sp.add_parser(
        "rates",
        help="compute angle rates from omega using a pseudoinverse mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    pvr.add_argument("mode", choices=["global", "local"])
    pvr.add_argument("seq", help=HELP_SEQ)
    pvr.add_argument("angles", type=str)
    pvr.add_argument("omega", type=str, help="Comma list of omega components.")
    pvr.add_argument("--frame", choices=["body", "space"], default="body")
    pvr.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret both angles and omega in degrees/deg/s; default is radians/rad/s.",
    )

    ps = sp.add_parser(
        "sphinx-skel",
        help="create a deploy-safe Sphinx documentation skeleton",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    ps.add_argument("dest", nargs="?", type=Path, default=Path("docs"), help="Destination docs directory.")
    ps.add_argument("--force", action="store_true", help="Overwrite existing Sphinx skeleton files.")

    return p


# -----------------------------------------------------------------------------
# Command runners
# -----------------------------------------------------------------------------
def _run_compose(args: argparse.Namespace) -> int:
    ang = _angles_arg(args.angles, args.degrees)
    Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
    print("R =\n", _fmt_ndarray(Robj.as_matrix()))
    if args.save:
        out = _resolve_out(args.save, "R.csv")
        core.save_R(out, Robj)
        print(f"saved to {out}")
    return 0


def _run_decompose(args: argparse.Namespace) -> int:
    if args.from_angles:
        ang = _angles_arg(args.from_angles, args.degrees)
        Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
    else:
        csv_path = _resolve_in(args.from_csv)
        M = np.loadtxt(csv_path, delimiter=",")
        Robj = R.from_matrix(M)
    a = core.decompose(Robj, args.seq, args.mode, degrees=args.degrees)
    unit = "deg" if args.degrees else "rad"
    print(f"angles ({unit}) = ", _fmt_ndarray(a))
    return 0


def _parse_points_arg(points: str | None) -> np.ndarray:
    """Read points from CSV or parse inline rows; default to identity basis."""
    if points and os.path.exists(_resolve_in(points)):
        return rio.read_points_csv(_resolve_in(points))
    if points:
        rows = [r for r in points.split("|") if r]
        return np.vstack([parse_floats(r) for r in rows])
    return np.eye(3)


def _run_transform(args: argparse.Namespace) -> int:
    ang = _angles_arg(args.angles, args.degrees)
    Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
    P = _parse_points_arg(args.points)
    Pg = core.transform_points(Robj, P)
    print("P' =\n", _fmt_ndarray(Pg))
    if args.save:
        out = _resolve_out(args.save, "points_out.csv")
        rio.write_points_csv(out, Pg)
        print(f"saved to {out}")
    return 0


def _run_passive(args: argparse.Namespace) -> int:
    ang = _angles_arg(args.angles, args.degrees)
    Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
    P = _parse_points_arg(args.points)
    Pb = core.passive_transform(Robj, P)
    print("coords in body =\n", _fmt_ndarray(Pb))
    return 0


def _run_repeat(args: argparse.Namespace) -> int:
    ang = _angles_arg(args.angles, args.degrees)
    Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
    Rm = core.repeat_rotation(Robj, args.m)
    print("R^m =\n", _fmt_ndarray(Rm.as_matrix()))
    return 0


def _run_align(args: argparse.Namespace) -> int:
    u = _vec_arg(args.u)
    Robj = core.align_body_x(u)
    print("R_align =\n", _fmt_ndarray(Robj.as_matrix()))
    return 0


def _run_check(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.from_csv:
        csv_path = _resolve_in(args.from_csv)
        M = np.loadtxt(csv_path, delimiter=",")
        Robj = R.from_matrix(M)
    else:
        if not args.from_angles:
            parser.error("provide --from-angles or --from-csv")
        ang = _angles_arg(args.from_angles, args.degrees)
        Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
    info = core.check_matrix(Robj)
    print("ok=", info.pop("ok"))
    for k, v in info.items():
        print(f"{k}: {v}")
    return 0


def _run_E(args: argparse.Namespace) -> int:
    seq = args.seq
    if seq == "zyx":
        E_seq, (a1, a2, a3), E_rpy, (phi, theta, psi) = cf.E_matrix_rpy_zyx_body(
            convention=args.convention
        )
        _print_sympy_matrix(
            E_seq,
            [str(a1), str(a2), str(a3)],
            "E_body for zyx (columns in seq-order [a1=psi, a2=theta, a3=phi]):\nomega = E(q) qdot",
        )
        if args.rpy_order:
            print()
            _print_sympy_matrix(
                E_rpy,
                [str(phi), str(theta), str(psi)],
                "E_body for zyx in RPY-order [phi, theta, psi] (columns match [phidot, thetadot, psidot]):\nomega = E_rpy(q) qdot",
            )
    else:
        E, (a1, a2, a3) = cf.E_matrix(seq, convention=args.convention, frame=args.frame)
        title = f"E_{args.frame} for {seq} (columns in seq-order [a1, a2, a3]):\nomega = E(q) qdot"
        _print_sympy_matrix(E, [str(a1), str(a2), str(a3)], title)
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    job_path = _resolve_in(args.file)
    job = rio.read_job(job_path)
    tasks = job.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("Batch file must contain a top-level 'tasks' list.")
    for i, t in enumerate(tasks, 1):
        print(f"\n--- Task {i}/{len(tasks)}: {t.get('cmd', '?')} ---")
        _run_one_task(t)
    return 0


def _run_angvel(args: argparse.Namespace) -> int:
    ang = _angles_arg(args.angles, args.degrees)
    rates = _angles_arg(args.rates, args.degrees)
    w = core.angvel_from_rates(
        args.seq, ang, rates, convention=args.mode, degrees=args.degrees, frame=args.frame
    )
    print("omega = ", _fmt_ndarray(w))
    return 0


def _run_rates(args: argparse.Namespace) -> int:
    ang = _angles_arg(args.angles, args.degrees)
    omg = _vec_arg(args.omega)
    qd = core.rates_from_angvel(
        args.seq, ang, omg, convention=args.mode, degrees=args.degrees, frame=args.frame
    )
    print("qdot = ", _fmt_ndarray(qd))
    return 0


def _run_sphinx_skel(args: argparse.Namespace) -> int:
    """Generate the Sphinx documentation skeleton."""
    written = _write_sphinx_skeleton(args.dest, force=args.force)

    dest = args.dest.expanduser().resolve()
    print(f"Sphinx skeleton ready: {dest}")
    if written:
        print("Written files:")
        for path in written:
            print(f"  {path}")
    else:
        print("No existing files were overwritten. Use --force to regenerate.")

    print("Support files ensured:")
    print(f"  {dest / '_static' / '.gitkeep'}")
    print(f"  {dest / '_templates' / '.gitkeep'}")
    return 0


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def cli(argv: List[str] | None = None) -> int:
    """Run the rotation-kinematics command-line interface."""
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    # Historical no-subcommand compatibility:
    #     python -m rotation_kinematics.rot_cli global zyx "30,20,10" --degrees
    # maps to:
    #     python -m rotation_kinematics.rot_cli compose global zyx "30,20,10" --degrees
    command_names = {
        "compose",
        "decompose",
        "transform",
        "passive",
        "repeat",
        "align",
        "check",
        "E",
        "batch",
        "angvel",
        "rates",
        "sphinx-skel",
        "-h",
        "--help",
    }
    if raw_argv and raw_argv[0] not in command_names:
        raw_argv = ["compose", *raw_argv]

    args = parser.parse_args(raw_argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd == "compose":
        return _run_compose(args)
    if args.cmd == "decompose":
        return _run_decompose(args)
    if args.cmd == "transform":
        return _run_transform(args)
    if args.cmd == "passive":
        return _run_passive(args)
    if args.cmd == "repeat":
        return _run_repeat(args)
    if args.cmd == "align":
        return _run_align(args)
    if args.cmd == "check":
        return _run_check(args, parser)
    if args.cmd == "E":
        return _run_E(args)
    if args.cmd == "batch":
        return _run_batch(args)
    if args.cmd == "angvel":
        return _run_angvel(args)
    if args.cmd == "rates":
        return _run_rates(args)
    if args.cmd == "sphinx-skel":
        return _run_sphinx_skel(args)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
