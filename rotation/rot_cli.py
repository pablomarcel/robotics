# =============================
# File: rotation/rot_cli.py
# =============================
from __future__ import annotations
import argparse
import os
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

from .rot_utils import ensure_dir, parse_floats
from .rot_design import VALID_ALL, HELP_SEQ
from . import rot_core as core
from . import rot_io as rio


OUT_DIR = os.path.join("rotation", "out")
IN_DIR  = os.path.join("rotation", "in")
ensure_dir(OUT_DIR)
ensure_dir(IN_DIR)


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


def cli(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rot-cli",
        description="Rotation Kinematics CLI (covers Eq. 2.1–2.363)",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    # compose
    pc = sp.add_parser("compose", help="Build a rotation matrix from a sequence and angles")
    pc.add_argument("mode", choices=["global","local"], help="Global/extrinsic or Local/intrinsic rotations")
    pc.add_argument("seq", help=HELP_SEQ)
    pc.add_argument("angles", type=str, help="comma list of three angles")
    pc.add_argument("--degrees", action="store_true", help="Angles are degrees (default radians)")
    pc.add_argument("--save", type=str, default=None, help="CSV path under rotation/out to save matrix")

    # decompose
    pd = sp.add_parser("decompose", help="Extract angles from a rotation matrix for a given sequence")
    pd.add_argument("mode", choices=["global","local"])
    pd.add_argument("seq", help=HELP_SEQ)
    g = pd.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-angles", type=str, help="If provided, first compose from these angles (same seq/mode)")
    g.add_argument("--from-csv", type=str, help="Path to 3x3 CSV matrix")
    pd.add_argument("--degrees", action="store_true")

    # transform (active)
    pt = sp.add_parser("transform", help="Active transform points: rG = R rB")
    pt.add_argument("mode", choices=["global","local"])
    pt.add_argument("seq", help=HELP_SEQ)
    pt.add_argument("angles", type=str)
    pt.add_argument("--degrees", action="store_true")
    pt.add_argument("--points", type=str, help="CSV of Nx3 points in rotation/in, else a csv list 'x;y;z|...'", default=None)
    pt.add_argument("--save", type=str, default=None, help="CSV path under rotation/out to save transformed points")

    # passive (change of basis)
    ppv = sp.add_parser("passive", help="Passive coordinate change: rB = R^T rG")
    ppv.add_argument("mode", choices=["global","local"])
    ppv.add_argument("seq", help=HELP_SEQ)
    ppv.add_argument("angles", type=str)
    ppv.add_argument("--degrees", action="store_true")
    ppv.add_argument("--points", type=str, default=None)

    # repeat
    pr = sp.add_parser("repeat", help="Repeat/Exponentiate a rotation (R^m)")
    pr.add_argument("mode", choices=["global","local"])
    pr.add_argument("seq", help=HELP_SEQ)
    pr.add_argument("angles", type=str)
    pr.add_argument("m", type=int)
    pr.add_argument("--degrees", action="store_true")

    # align
    pa = sp.add_parser("align", help="Build R that aligns body x-axis to vector u and y into XY-plane")
    pa.add_argument("u", type=str, help="comma list for target vector u")

    # check
    pk = sp.add_parser("check", help="Check orthogonality/det=1 (2.285–2.293)")
    pk.add_argument("--from-angles", type=str, help="compose first")
    pk.add_argument("--mode", choices=["global","local"], default="global")
    pk.add_argument("--seq", default="zyx")
    pk.add_argument("--degrees", action="store_true")
    pk.add_argument("--from-csv", type=str, help="matrix CSV path")

    # angvel from rates
    pv = sp.add_parser(
        "angvel",
        help="Compute ω from angle rates for any sequence "
             "(2.169/2.170/2.211–2.222/2.243–2.249)",
    )
    pv.add_argument("mode", choices=["global","local"])
    pv.add_argument("seq", help=HELP_SEQ)
    pv.add_argument("angles", type=str)
    pv.add_argument(
        "rates",
        type=str,
        help="comma list of angle rates (units follow --degrees: deg/s if set, else rad/s)",
    )
    pv.add_argument("--frame", choices=["body","space"], default="body")
    pv.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret BOTH angles and rates in degrees/deg·s⁻¹ (default radians/rad·s⁻¹)",
    )

    # rates from angvel
    pvr = sp.add_parser(
        "rates",
        help="Compute angle rates from ω via pseudoinverse mapping (handles singularities)",
    )
    pvr.add_argument("mode", choices=["global","local"])
    pvr.add_argument("seq", help=HELP_SEQ)
    pvr.add_argument("angles", type=str)
    pvr.add_argument(
        "omega",
        type=str,
        help="comma list of ω components (units follow --degrees: deg/s if set, else rad/s)",
    )
    pvr.add_argument("--frame", choices=["body","space"], default="body")
    pvr.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret BOTH angles and ω in degrees/deg·s⁻¹ (default radians/rad·s⁻¹)",
    )

    args = p.parse_args(argv)

    if args.cmd == 'compose':
        ang = _angles_arg(args.angles, args.degrees)
        Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        print("R =\n", np.array2string(Robj.as_matrix(), formatter={'float_kind':lambda x:f"{x: .6f}"}))
        if args.save:
            out = os.path.join(OUT_DIR, args.save)
            core.save_R(out, Robj)
            print(f"saved to {out}")
        return 0

    if args.cmd == 'decompose':
        if args.from_angles:
            ang = _angles_arg(args.from_angles, args.degrees)
            Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        else:
            M = np.loadtxt(args.from_csv, delimiter=",")
            Robj = R.from_matrix(M)
        a = core.decompose(Robj, args.seq, args.mode, degrees=args.degrees)
        unit = "deg" if args.degrees else "rad"
        print(f"angles ({unit}) = ", np.array2string(a, formatter={'float_kind':lambda x:f"{x: .6f}"}))
        return 0

    if args.cmd == 'transform':
        ang = _angles_arg(args.angles, args.degrees)
        Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        if args.points and os.path.exists(os.path.join(IN_DIR, args.points)):
            P = rio.read_points_csv(os.path.join(IN_DIR, args.points))
        elif args.points:
            rows = [r for r in args.points.split('|') if r]
            P = np.vstack([parse_floats(r) for r in rows])
        else:
            P = np.eye(3)
        Pg = core.transform_points(Robj, P)
        print("P' =\n", np.array2string(Pg, formatter={'float_kind':lambda x:f"{x: .6f}"}))
        if args.save:
            out = os.path.join(OUT_DIR, args.save)
            rio.write_points_csv(out, Pg)
            print(f"saved to {out}")
        return 0

    if args.cmd == 'passive':
        ang = _angles_arg(args.angles, args.degrees)
        Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        if args.points and os.path.exists(os.path.join(IN_DIR, args.points)):
            P = rio.read_points_csv(os.path.join(IN_DIR, args.points))
        elif args.points:
            rows = [r for r in args.points.split('|') if r]
            P = np.vstack([parse_floats(r) for r in rows])
        else:
            P = np.eye(3)
        Pb = core.passive_transform(Robj, P)
        print("coords in body =\n", np.array2string(Pb, formatter={'float_kind':lambda x:f"{x: .6f}"}))
        return 0

    if args.cmd == 'repeat':
        ang = _angles_arg(args.angles, args.degrees)
        Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        Rm = core.repeat_rotation(Robj, args.m)
        print("R^m =\n", np.array2string(Rm.as_matrix(), formatter={'float_kind':lambda x:f"{x: .6f}"}))
        return 0

    if args.cmd == 'align':
        u = _vec_arg(args.u)
        Robj = core.align_body_x(u)
        print("R_align =\n", np.array2string(Robj.as_matrix(), formatter={'float_kind':lambda x:f"{x: .6f}"}))
        return 0

    if args.cmd == 'check':
        if args.from_csv:
            M = np.loadtxt(args.from_csv, delimiter=",")
            Robj = R.from_matrix(M)
        else:
            if not args.from_angles:
                p.error("provide --from-angles or --from-csv")
            ang = _angles_arg(args.from_angles, args.degrees)
            Robj = core.build_matrix(args.mode, args.seq, ang, degrees=args.degrees)
        info = core.check_matrix(Robj)
        print("ok=", info.pop('ok'))
        for k, v in info.items():
            print(f"{k}: {v}")
        return 0

    if args.cmd == 'angvel':
        ang = _angles_arg(args.angles, args.degrees)
        rates = _angles_arg(args.rates, args.degrees)  # units follow --degrees
        w = core.angvel_from_rates(
            args.seq, ang, rates,
            convention=args.mode, degrees=args.degrees, frame=args.frame
        )
        print("ω = ", np.array2string(w, formatter={'float_kind':lambda x:f"{x: .6f}"}))
        return 0

    if args.cmd == 'rates':
        ang = _angles_arg(args.angles, args.degrees)
        omg = _vec_arg(args.omega)  # units follow --degrees
        qd = core.rates_from_angvel(
            args.seq, ang, omg,
            convention=args.mode, degrees=args.degrees, frame=args.frame
        )
        print("qdot = ", np.array2string(qd, formatter={'float_kind':lambda x:f"{x: .6f}"}))
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
