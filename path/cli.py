from __future__ import annotations
import argparse, sys, json
import numpy as np
from .app import PathPlannerApp
from .core import BoundaryConditions

def build_parser():
    p = argparse.ArgumentParser("path-cli", description="Path planning CLI (12.1–12.301)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # cubic/quintic/septic
    for name in ("cubic","quintic","septic"):
        sp = sub.add_parser(name, help=f"{name} polynomial with textbook BCs")
        sp.add_argument("--t0", type=float, required=True)
        sp.add_argument("--tf", type=float, required=True)
        sp.add_argument("--q0", type=float, required=True)
        sp.add_argument("--qf", type=float, required=True)
        sp.add_argument("--qd0", type=float, default=0.0)
        sp.add_argument("--qdf", type=float, default=0.0)
        sp.add_argument("--qdd0", type=float, default=0.0)
        sp.add_argument("--qddf", type=float, default=0.0)
        sp.add_argument("--samples", type=int, default=200)
        sp.add_argument("--out", type=str, default=f"{name}.csv")

    # lspb
    sp = sub.add_parser("lspb", help="LSPB trapezoidal/triangular time-law")
    sp.add_argument("--t0","--t-start", dest="t0", type=float, required=True)
    sp.add_argument("--tf","--t-end", dest="tf", type=float, required=True)
    sp.add_argument("--q0", type=float, required=True)
    sp.add_argument("--qf", type=float, required=True)
    sp.add_argument("--vmax", type=float)
    sp.add_argument("--amax", type=float)
    sp.add_argument("--samples", type=int, default=200)
    sp.add_argument("--out", type=str, default="lspb.csv")

    # 2R IK along line/circle via JSON spec
    sp = sub.add_parser("ik-2r", help="Follow Cartesian path with 2R IK")
    sp.add_argument("--spec", type=str, required=True, help="JSON in path/in (see docs)")
    sp.add_argument("--out", type=str, default="ik2r.csv")

    # rotation angle-axis
    sp = sub.add_parser("rot", help="Angle-axis rotation path between two rotation matrices")
    sp.add_argument("--R0", type=str, required=True, help="JSON filename with 3x3 R0")
    sp.add_argument("--Rf", type=str, required=True, help="JSON filename with 3x3 Rf")
    sp.add_argument("--samples", type=int, default=50)
    sp.add_argument("--out", type=str, default="rot.json")
    return p

def main(argv=None):
    argv = argv or sys.argv[1:]
    args = build_parser().parse_args(argv)
    app = PathPlannerApp()

    if args.cmd in ("cubic","quintic","septic","lspb"):
        bc = BoundaryConditions(args.t0, args.tf, args.q0, args.qf, args.qd0, args.qdf, args.qdd0, args.qddf)
        traj = getattr(app, args.cmd)(bc) if args.cmd!="lspb" else app.lspb(bc, vmax=args.vmax, amax=args.amax)
        t = np.linspace(args.t0, args.tf, args.samples)
        samp = app.sample_1d(traj, t)
        app.io.write_csv(args.out, t=samp.t, q=samp.q, qd=samp.qd, qdd=samp.qdd)

    elif args.cmd == "ik-2r":
        spec = app.io.read_json(args.spec)
        arm = app.planar2r(spec["l1"], spec["l2"], spec.get("elbow","up"))
        t = np.linspace(spec["t0"], spec["tf"], spec.get("samples",200))
        if spec["path"]["type"] == "line":
            X = np.linspace(spec["path"]["x0"], spec["path"]["x1"], t.size)
            Y = np.linspace(spec["path"]["y0"], spec["path"]["y1"], t.size)
        elif spec["path"]["type"] == "circle":
            cx,cy,R = spec["path"]["cx"], spec["path"]["cy"], spec["path"]["R"]
            s = np.linspace(spec["path"]["s0"], spec["path"]["s1"], t.size)
            X = cx + R*np.cos(s); Y = cy + R*np.sin(s)
        th1, th2 = arm.ik(X, Y)
        app.io.write_csv(args.out, t=t, X=X, Y=Y, th1=th1, th2=th2)

    elif args.cmd == "rot":
        import json
        R0 = np.array(app.io.read_json(args.R0)["R"])
        Rf = np.array(app.io.read_json(args.Rf)["R"])
        path = app.angle_axis_path(R0, Rf)
        s = np.linspace(0, 1, args.samples)
        Rseq = path.R(s).tolist()
        app.io.write_json(args.out, {"R": Rseq})
    else:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
