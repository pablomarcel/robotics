from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from .apis import DynamicsAPI
from .design import DHChainBuilder
from .core import State
from .io import IOMgr, IOConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="robot-dyn", description="Robot dynamics CLI (OOP + TDD)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 2R quick-run
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
    q2r.add_argument("--out", type=Path, default=Path("robot/out/2r_result.json"))

    # generic from YAML
    gy = sub.add_parser("from-yaml", help="Load model from robot/in YAML and compute dynamics")
    gy.add_argument("name", type=str, help="YAML filename in robot/in")
    gy.add_argument("--engine", choices=["sympy", "pinocchio"], default="sympy")
    gy.add_argument("--q", type=float, nargs='+', required=True)
    gy.add_argument("--qd", type=float, nargs='+', required=True)
    gy.add_argument("--qdd", type=float, nargs='+', default=None)
    gy.add_argument("--g", type=float, default=9.81)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    io = IOMgr(IOConfig(Path("robot/in"), Path("robot/out")))

    if args.cmd == "planar2r":
        model, _ = DHChainBuilder.planar_2r(args.l1, args.l2, args.m1, args.m2)
        api = DynamicsAPI(engine=args.engine)
        state = State(q=np.array(args.q), qd=np.array(args.qd), qdd=np.array(args.qdd))
        res = api.run(model, state, gravity=args.g)
        io.save_json(args.out.name, {k: (v.tolist() if v is not None else None) for k, v in res.items()})
        print(f"Saved results to {args.out}")
        return 0

    if args.cmd == "from-yaml":
        model = io.model_from_yaml(args.name)
        api = DynamicsAPI(engine=args.engine)
        q = np.array(args.q); qd = np.array(args.qd)
        qdd = np.array(args.qdd) if args.qdd is not None else None
        res = api.run(model, State(q, qd, qdd), gravity=args.g)
        io.save_json(Path(args.name).with_suffix('.json').name,
                     {k: (v.tolist() if v is not None else None) for k, v in res.items()})
        return 0

    return 1

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())