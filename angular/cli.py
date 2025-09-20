from __future__ import annotations
import argparse, sys, numpy as np
from .apis import AngularAPI

def numpy3(arg: str):
    vals = [float(x) for x in arg.split(",")]
    if len(vals) != 3: raise argparse.ArgumentTypeError("Expect 3 comma-separated floats.")
    return np.array(vals, float)

def numpy9(arg: str):
    vals = [float(x) for x in arg.split(",")]
    if len(vals) != 9: raise argparse.ArgumentTypeError("Expect 9 comma-separated floats.")
    return np.array(vals, float).reshape(3,3)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="angular", description="Angular velocity toolkit CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # from-euler
    s = sub.add_parser("from-euler", help="Build rotation from Euler angles")
    s.add_argument("--order", required=True, help="e.g., ZYX")
    s.add_argument("--angles", required=True, type=numpy3, help="phi,theta,psi (rad)")
    s.add_argument("--save", help="filename.npy in angular/out")

    # omega-from-Rdot
    s = sub.add_parser("omega-from-Rdot", help="Compute ω̃ = Rdot R^T")
    s.add_argument("--R", required=True, type=numpy9)
    s.add_argument("--Rdot", required=True, type=numpy9)

    # Rdot-from-omega
    s = sub.add_parser("Rdot-from-omega", help="Compute Rdot = ω̃ R")
    s.add_argument("--R", required=True, type=numpy9)
    s.add_argument("--omega", required=True, type=numpy3)

    # velocity-matrix
    s = sub.add_parser("velocity-matrix", help="Compute V = Tdot T^{-1}")
    s.add_argument("--R", required=True, type=numpy9)
    s.add_argument("--d", required=True, type=numpy3)
    s.add_argument("--Rdot", required=True, type=numpy9)
    s.add_argument("--ddot", required=True, type=numpy3)

    # rigid-point-velocity
    s = sub.add_parser("rigid-pt-vel", help="Compute vP = ω×(rP-dB)+dBdot")
    s.add_argument("--omega", required=True, type=numpy3)
    s.add_argument("--rP", required=True, type=numpy3)
    s.add_argument("--dB", required=True, type=numpy3)
    s.add_argument("--dBdot", required=True, type=numpy3)

    # screw-from-twist
    s = sub.add_parser("screw-from-twist", help="Decompose 6×1 twist into axis/moment/pitch")
    s.add_argument("--twist", required=True, type=lambda s: np.array([float(x) for x in s.split(",")], float))
    return p

def main(argv=None):
    api = AngularAPI()
    ns = build_parser().parse_args(argv)
    if ns.cmd == "from-euler":
        R = api.rotation_from_euler(ns.order, tuple(ns.angles)).R
        if ns.save: api.persist_rotation(ns.save, R)
        print(R)
    elif ns.cmd == "omega-from-Rdot":
        print(api.omega_from_Rdot(ns.R, ns.Rdot))
    elif ns.cmd == "Rdot-from-omega":
        print(api.Rdot_from_omega(ns.R, ns.omega))
    elif ns.cmd == "velocity-matrix":
        print(api.velocity_matrix(ns.R, ns.d, ns.Rdot, ns.ddot))
    elif ns.cmd == "rigid-pt-vel":
        print(api.rigid_point_velocity(ns.omega, ns.rP, ns.dB, ns.dBdot))
    elif ns.cmd == "screw-from-twist":
        screw = api.screw_from_twist(ns.twist)
        print({"s": screw.s.tolist(), "m": screw.m.tolist(), "pitch": float(screw.pitch)})

if __name__ == "__main__":
    main()
