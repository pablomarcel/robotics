# =============================
# File: rotation_kinematics/rot_cli.py  (upgraded)
# =============================
from __future__ import annotations
import argparse
import os
from typing import List, Any, Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .rot_utils import ensure_dir, parse_floats
from .rot_design import VALID_ALL, HELP_SEQ
from . import rot_core as core
from . import rot_io as rio
from . import rot_closedform as cf

# ---------------------------------------------------------
# Defaults and I/O conveniences ("dirty trick" upgrade)
# ---------------------------------------------------------
OUT_DIR = os.path.join("rotation_kinematics", "out")
IN_DIR  = os.path.join("rotation_kinematics", "in")
ensure_dir(OUT_DIR); ensure_dir(IN_DIR)

def _resolve_in(path_like: str) -> str:
    """
    Resolve an input file by searching rotation_kinematics/in, then rotation_kinematics/out,
    then using the provided path_planning as-is.
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
    Resolve an output path_planning. If 'path_like' is None or a bare filename,
    write under rotation_kinematics/out. Parents are created as needed.
    """
    if not path_like:
        out = os.path.join(OUT_DIR, default_name)
    else:
        # If just a bare name (no dir components), drop it into OUT_DIR
        if os.path.dirname(path_like) == "":
            out = os.path.join(OUT_DIR, path_like)
        else:
            out = path_like
    ensure_dir(os.path.dirname(out))
    return out

# ---------------------------------------------------------
# Small parse/print helpers
# ---------------------------------------------------------
def _angles_arg(s: str, degrees: bool) -> List[float]:
    vals = parse_floats(s)
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("need 3 comma-separated values")
    return vals

def _vec_arg(s: str) -> List[float]:
    vals = parse_floats(s)
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("need 3 comma-separated values")
    return vals

def _fmt_ndarray(arr: np.ndarray) -> str:
    return np.array2string(arr, formatter={'float_kind':lambda x:f"{x: .6f}"})

def _print_sympy_matrix(M, names: List[str], title: str):
    # Pretty print without the "Matrix(...)" wrapper
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

# ---------------------------------------------------------
# Batch task exec
# ---------------------------------------------------------
def _run_one_task(t: Dict[str, Any]) -> None:
    """Execute a single batch task dict."""
    cmd = t.get("cmd")
    if cmd == "compose":
        Robj = core.build_matrix(t["mode"], t["seq"], t["angles"], degrees=bool(t.get("degrees", False)))
        print(_fmt_ndarray(Robj.as_matrix()))
        if "save" in t:
            out = _resolve_out(t["save"], "R.csv")
            core.save_R(out, Robj); print(f"saved -> {out}")
    elif cmd == "transform":
        Robj = core.build_matrix(t["mode"], t["seq"], t["angles"], degrees=bool(t.get("degrees", False)))
        if "points_file" in t:
            pf = _resolve_in(t["points_file"])
            P = rio.read_points_csv(pf)
        else:
            P = np.array(t.get("points", []), dtype=float)
            if P.size == 0: P = np.eye(3)
        Pg = core.transform_points(Robj, P)
        print(_fmt_ndarray(Pg))
        if "save" in t:
            out = _resolve_out(t["save"], "points_out.csv")
            rio.write_points_csv(out, Pg); print(f"saved -> {out}")
    elif cmd == "passive":
        Robj = core.build_matrix(t["mode"], t["seq"], t["angles"], degrees=bool(t.get("degrees", False)))
        P = np.array(t.get("points", []), dtype=float)
        if P.size == 0: P = np.eye(3)
        Pb = core.passive_transform(Robj, P); print(_fmt_ndarray(Pb))
    elif cmd == "angvel":
        w = core.angvel_from_rates(
            t["seq"], t["angles"], t["rates"],
            convention=t["mode"], degrees=bool(t.get("degrees", False)),
            frame=t.get("frame","body"),
        )
        print(_fmt_ndarray(w))
    elif cmd == "rates":
        qdot = core.rates_from_angvel(
            t["seq"], t["angles"], t["omega"],
            convention=t["mode"], degrees=bool(t.get("degrees", False)),
            frame=t.get("frame","body"),
        )
        print(_fmt_ndarray(qdot))
    elif cmd == "repeat":
        Robj = core.build_matrix(t["mode"], t["seq"], t["angles"], degrees=bool(t.get("degrees", False)))
        Rm = core.repeat_rotation(Robj, int(t["m"])); print(_fmt_ndarray(Rm.as_matrix()))
    elif cmd == "align":
        Robj = core.align_body_x(t["u"]); print(_fmt_ndarray(Robj.as_matrix()))
    else:
        raise ValueError(f"Unknown batch cmd: {cmd}")

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def cli(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rot-cli",
        description=(
            "Rotation Kinematics CLI (Eq. 2.1–2.363 + extras) — "
            "now with friendlier in/out handling and a Sphinx skeleton generator"
        ),
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    # compose
    pc = sp.add_parser("compose", help="Build a rotation_kinematics matrix from a sequence and angles")
    pc.add_argument("mode", choices=["global","local"], help="Global/extrinsic or Local/intrinsic rotations")
    pc.add_argument("seq", help=HELP_SEQ)
    pc.add_argument("angles", type=str, help="comma list of three angles")
    pc.add_argument("--degrees", action="store_true", help="Angles are degrees (default radians)")
    pc.add_argument("--save", type=str, default=None, help="CSV filename or path_planning to save matrix (defaults to rotation_kinematics/out)")

    # decompose
    pd = sp.add_parser("decompose", help="Extract angles from a rotation_kinematics matrix for a given sequence")
    pd.add_argument("mode", choices=["global","local"])
    pd.add_argument("seq", help=HELP_SEQ)
    g = pd.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-angles", type=str, help="If provided, first compose from these angles (same seq/mode)")
    g.add_argument("--from-csv", type=str, help="Matrix CSV filename (resolved via rotation_kinematics/in → rotation_kinematics/out → literal)")
    pd.add_argument("--degrees", action="store_true")

    # transform
    pt = sp.add_parser("transform", help="Active transform points: rG = R rB")
    pt.add_argument("mode", choices=["global","local"])
    pt.add_argument("seq", help=HELP_SEQ)
    pt.add_argument("angles", type=str)
    pt.add_argument("--degrees", action="store_true")
    pt.add_argument("--points", type=str, help="CSV of Nx3 points (resolved via in/out) or rows 'x;y;z|...'", default=None)
    pt.add_argument("--save", type=str, default=None, help="CSV filename/path_planning to save transformed points (defaults to rotation_kinematics/out)")

    # passive
    ppv = sp.add_parser("passive", help="Passive coordinate change: rB = R^T rG")
    ppv.add_argument("mode", choices=["global","local"])
    ppv.add_argument("seq", help=HELP_SEQ)
    ppv.add_argument("angles", type=str)
    ppv.add_argument("--degrees", action="store_true")
    ppv.add_argument("--points", type=str, default=None, help="CSV filename (resolved via in/out) or inline rows")

    # repeat
    pr = sp.add_parser("repeat", help="Repeat/Exponentiate a rotation_kinematics (R^m)")
    pr.add_argument("mode", choices=["global","local"])
    pr.add_argument("seq", help=HELP_SEQ)
    pr.add_argument("angles", type=str)
    pr.add_argument("m", type=int)
    pr.add_argument("--degrees", action="store_true")

    # align
    pa = sp.add_parser("align", help="Build R that aligns body x-axis to vector u and y into XY-plane")
    pa.add_argument("u", type=str, help="comma list for target vector u")

    # check
    pk = sp.add_parser("check", help="Check orthogonality/det=1")
    pk.add_argument("--from-angles", type=str, help="compose first")
    pk.add_argument("--mode", choices=["global","local"], default="global")
    pk.add_argument("--seq", default="zyx")
    pk.add_argument("--degrees", action="store_true")
    pk.add_argument("--from-csv", type=str, help="matrix CSV filename (resolved via in/out/literal)")

    # Closed-form E matrices
    pe = sp.add_parser("E", help="Closed-form E(q) such that ω = E(q) q̇")
    pe.add_argument("seq", choices=["zyx","zxz","zyz"], help="RPY ZYX or proper Euler ZXZ/ZYZ")
    pe.add_argument("--convention", choices=["local","global"], default="local", help="Angle convention")
    pe.add_argument("--frame", choices=["body","space"], default="body", help="Frame of ω")
    pe.add_argument("--rpy-order", action="store_true",
                    help="For zyx only: also print RPY order [φ,θ,ψ] columns instead of seq-order [ψ,θ,φ]")

    # Batch jobs
    pb = sp.add_parser("batch", help="Run a JSON/YAML job file with multiple tasks")
    pb.add_argument("file", type=str, help="Filename under rotation_kinematics/in (resolver also checks rotation_kinematics/out and literal)")

    # angvel from rates
    pv = sp.add_parser("angvel", help="Compute ω from angle rates for any sequence")
    pv.add_argument("mode", choices=["global","local"])
    pv.add_argument("seq", help=HELP_SEQ)
    pv.add_argument("angles", type=str)
    pv.add_argument("rates", type=str, help="comma list of angle rates (units follow --degrees)")
    pv.add_argument("--frame", choices=["body","space"], default="body")
    pv.add_argument("--degrees", action="store_true",
                    help="Interpret BOTH angles and rates in degrees/deg·s⁻¹ (default radians/rad·s⁻¹)")

    # rates from angvel
    pvr = sp.add_parser("rates", help="Compute angle rates from ω (pseudoinverse mapping)")
    pvr.add_argument("mode", choices=["global","local"])
    pvr.add_argument("seq", help=HELP_SEQ)
    pvr.add_argument("angles", type=str)
    pvr.add_argument("omega", type=str, help="comma list of ω components (units follow --degrees)")
    pvr.add_argument("--frame", choices=["body","space"], default="body")
    pvr.add_argument("--degrees", action="store_true",
                     help="Interpret BOTH angles and ω in degrees/deg·s⁻¹ (default radians/rad·s⁻¹)")

    # Sphinx skeleton
    ps = sp.add_parser("sphinx-skel", help="Create a minimal Sphinx docs skeleton (rotation_kinematics/docs)")
    ps.add_argument("dest", nargs="?", default="docs", help="Destination directory (default: docs)")
    ps.add_argument("--force", action="store_true", help="Overwrite existing files if present")

    args = p.parse_args(argv)

    if args.cmd == 'compose':
        ang = _angles_arg(args.angles, args.degrees)
        Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        print("R =\n", _fmt_ndarray(Robj.as_matrix()))
        if args.save:
            out = _resolve_out(args.save, "R.csv")
            core.save_R(out, Robj); print(f"saved to {out}")
        return 0

    if args.cmd == 'decompose':
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

    if args.cmd == 'transform':
        ang = _angles_arg(args.angles, args.degrees)
        Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        if args.points and os.path.exists(_resolve_in(args.points)):
            P = rio.read_points_csv(_resolve_in(args.points))
        elif args.points:
            rows = [r for r in args.points.split('|') if r]
            P = np.vstack([parse_floats(r) for r in rows])
        else:
            P = np.eye(3)
        Pg = core.transform_points(Robj, P)
        print("P' =\n", _fmt_ndarray(Pg))
        if args.save:
            out = _resolve_out(args.save, "points_out.csv")
            rio.write_points_csv(out, Pg); print(f"saved to {out}")
        return 0

    if args.cmd == 'passive':
        ang = _angles_arg(args.angles, args.degrees)
        Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        if args.points and os.path.exists(_resolve_in(args.points)):
            P = rio.read_points_csv(_resolve_in(args.points))
        elif args.points:
            rows = [r for r in args.points.split('|') if r]
            P = np.vstack([parse_floats(r) for r in rows])
        else:
            P = np.eye(3)
        Pb = core.passive_transform(Robj, P)
        print("coords in body =\n", _fmt_ndarray(Pb))
        return 0

    if args.cmd == 'repeat':
        ang = _angles_arg(args.angles, args.degrees)
        Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        Rm = core.repeat_rotation(Robj, args.m)
        print("R^m =\n", _fmt_ndarray(Rm.as_matrix()))
        return 0

    if args.cmd == 'align':
        u = _vec_arg(args.u)
        Robj = core.align_body_x(u)
        print("R_align =\n", _fmt_ndarray(Robj.as_matrix()))
        return 0

    if args.cmd == 'check':
        if args.from_csv:
            csv_path = _resolve_in(args.from_csv)
            M = np.loadtxt(csv_path, delimiter=",")
            Robj = R.from_matrix(M)
        else:
            if not args.from_angles:
                p.error("provide --from-angles or --from-csv")
            ang = _angles_arg(args.from_angles, args.degrees)
            Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        info = core.check_matrix(Robj)
        print("ok=", info.pop('ok'))
        for k,v in info.items(): print(f"{k}: {v}")
        return 0

    if args.cmd == 'E':
        seq = args.seq
        if seq == 'zyx':
            E_seq, (a1,a2,a3), E_rpy, (phi,theta,psi) = cf.E_matrix_rpy_zyx_body(convention=args.convention)
            _print_sympy_matrix(E_seq, [str(a1), str(a2), str(a3)],
                                "E_body for zyx (columns in seq-order [a1=ψ, a2=θ, a3=φ]):\nω = E(q) q̇")
            if args.rpy_order:
                print()
                _print_sympy_matrix(E_rpy, [str(phi), str(theta), str(psi)],
                                    "E_body for zyx in RPY-order [φ, θ, ψ] (columns match [φ̇, θ̇, ψ̇]):\nω = E_rpy(q) q̇")
        else:
            E, (a1,a2,a3) = cf.E_matrix(seq, convention=args.convention, frame=args.frame)
            title = f"E_{args.frame} for {seq} (columns in seq-order [a1, a2, a3]):\nω = E(q) q̇"
            _print_sympy_matrix(E, [str(a1), str(a2), str(a3)], title)
        return 0

    if args.cmd == 'batch':
        job_path = _resolve_in(args.file)
        job = rio.read_job(job_path)
        tasks = job.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            raise SystemExit("Batch file must contain a top-level 'tasks' list.")
        for i, t in enumerate(tasks, 1):
            print(f"\n--- Task {i}/{len(tasks)}: {t.get('cmd','?')} ---")
            _run_one_task(t)
        return 0

    if args.cmd == 'angvel':
        ang = _angles_arg(args.angles, args.degrees)
        rates = _angles_arg(args.rates, args.degrees)  # units follow --degrees
        w = core.angvel_from_rates(args.seq, ang, rates, convention=args.mode, degrees=args.degrees, frame=args.frame)
        print("ω = ", _fmt_ndarray(w))
        return 0

    if args.cmd == 'rates':
        ang = _angles_arg(args.angles, args.degrees)
        omg = _vec_arg(args.omega)  # units follow --degrees
        qd = core.rates_from_angvel(args.seq, ang, omg, convention=args.mode, degrees=args.degrees, frame=args.frame)
        print("qdot = ", _fmt_ndarray(qd))
        return 0

    if args.cmd == 'sphinx-skel':
        dest = args.dest
        out_dir = os.path.abspath(dest)
        os.makedirs(out_dir, exist_ok=True)

        conf = (
            '# Generated by rotation_kinematics.rot_cli\n'
            'project = "rotation_kinematics"\n'
            'extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]\n'
            'templates_path = ["_templates"]\n'
            'exclude_patterns = []\n'
            'html_theme = "furo"\n'
        )
        index = (
            ".. rotation_kinematics documentation master file\n\n"
            "Welcome to rotation_kinematics's docs!\n"
            "===========================\n\n"
            ".. toctree::\n"
            "   :maxdepth: 2\n"
            "   :caption: Contents:\n\n"
            "   api\n"
        )
        api = (
            "API Reference\n"
            "=============\n\n"
            ".. automodule:: rotation_kinematics.rot_core\n   :members:\n   :undoc-members:\n\n"
            ".. automodule:: rotation_kinematics.rot_io\n   :members:\n   :undoc-members:\n\n"
            ".. automodule:: rotation_kinematics.rot_design\n   :members:\n   :undoc-members:\n\n"
            ".. automodule:: rotation_kinematics.rot_closedform\n   :members:\n   :undoc-members:\n\n"
        )
        makefile = (
            "# Minimal Sphinx Makefile\n"
            ".PHONY: html clean\n"
            "html:\n\t+sphinx-build -b html . _build/html\n"
            "clean:\n\t+rm -rf _build\n"
        )

        def _write(p: str, text: str) -> None:
            if os.path.exists(p) and not args.force:
                return
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)

        os.makedirs(os.path.join(out_dir, "_templates"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "_static"), exist_ok=True)
        _write(os.path.join(out_dir, "conf.py"), conf)
        _write(os.path.join(out_dir, "index.rst"), index)
        _write(os.path.join(out_dir, "api.rst"), api)
        _write(os.path.join(out_dir, "Makefile"), makefile)
        print(out_dir)
        return 0

    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
