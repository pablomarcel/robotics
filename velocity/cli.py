# velocity/cli.py
"""
Command-line interface for the Velocity Kinematics Toolkit.

Design goals
------------
- Zero non-stdlib dependencies (pure argparse).
- Thin, testable commands that delegate to `VelocityAPI`.
- Exhaustive yet ergonomic flags (CSV/JSON inputs, files or literals).
- No side effects unless explicitly requested (stdout or --out).

Examples
--------
# Forward kinematics on a DH YAML
python -m velocity.cli fk velocity/in/planar2r.yml --q 0.3,0.2 --out velocity/out/fk.json

# Geometric Jacobian and resolved rates
python -m velocity.cli jacobian velocity/in/arm.urdf --q 0.1,0.2,0.3
python -m velocity.cli resolved-rates velocity/in/arm.yml --q 0.1,0.2 --xdot 0.1,0,0,0,0,0

# Newton–Raphson IK: position + ZYX Euler
python -m velocity.cli newton-ik velocity/in/arm.yml --q0 0,0,0 --p 0.5,0.1,0.2 --euler zyx 10,20,0 --max-iter 100

# Linear algebra exercises (chapter 8.5+)
python -m velocity.cli lu-solve --A "[[2,1],[1,3]]" --b "[1,2]"
python -m velocity.cli lu-inv   --A velocity/in/matrix.json

# Class diagram (optional pylint presence)
python -m velocity.cli diagram --out velocity/out
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .apis import VelocityAPI, RobotSpec, APIError
from . import design


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _read_json_array(src: str | Path) -> Any:
    p = Path(src)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _maybe_json_or_csv(arg: Optional[str]) -> Optional[List[float]]:
    if arg is None:
        return None
    arg = arg.strip()
    if not arg:
        return None
    # JSON array?
    if (arg.startswith("[") and arg.endswith("]")) or (arg.startswith("(") and arg.endswith(")")):
        return list(np.asarray(json.loads(arg), dtype=float).ravel())
    # File path?
    p = Path(arg)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return list(np.asarray(data, dtype=float).ravel())
    # CSV literals
    return _parse_csv_floats(arg)

def _dump(obj: Any) -> str:
    def _default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return str(o)
    return json.dumps(obj, indent=2, default=_default)

def _write_or_print(payload: Mapping[str, Any] | Sequence[Any] | Any, out: Optional[str | Path]) -> int:
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(_dump(payload), encoding="utf-8")
    else:
        print(_dump(payload))
    return 0


# --------------------------------------------------------------------------- #
# CLI object (for OOP + TDD)
# --------------------------------------------------------------------------- #

@dataclass
class VelocityCLI:
    api: VelocityAPI = VelocityAPI()

    # -------- parser construction (isolated for unit tests) --------

    def build_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            prog="velocity",
            description="Velocity Kinematics Toolkit CLI",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        sub = p.add_subparsers(dest="cmd", required=True)

        # fk
        fk = sub.add_parser("fk", help="Forward kinematics")
        fk.add_argument("robot", help="Path to DH (.yml/.yaml/.json) or URDF (.urdf/.xml)")
        fk.add_argument("--q", required=True, help="Joint vector as CSV/JSON or path to JSON")
        fk.add_argument("--out", help="Write JSON output to path")
        fk.set_defaults(func=self._cmd_fk)

        # jacobian
        ja = sub.add_parser("jacobian", help="Geometric Jacobian J")
        ja.add_argument("robot")
        ja.add_argument("--q", required=True)
        ja.add_argument("--out")
        ja.set_defaults(func=self._cmd_jacobian)

        # analytic jacobian
        jaa = sub.add_parser("jacobian-analytic", help="Analytic Jacobian J_A (Euler-rate mapping)")
        jaa.add_argument("robot")
        jaa.add_argument("--q", required=True)
        jaa.add_argument("--euler", default="ZYX", help="Euler sequence (e.g., ZYX, ZXZ)")
        jaa.add_argument("--out")
        jaa.set_defaults(func=self._cmd_jacobian_analytic)

        # resolved rates
        rr = sub.add_parser("resolved-rates", help="Inverse velocity qdot from Xdot = J qdot")
        rr.add_argument("robot")
        rr.add_argument("--q", required=True)
        rr.add_argument("--xdot", required=True, help="Task-space rates [vx,vy,vz, wx,wy,wz]")
        rr.add_argument("--damping", type=float, default=None, help="Damped least squares lambda")
        rr.add_argument("--weights", help="Optional task weights (CSV/JSON)")
        rr.add_argument("--out")
        rr.set_defaults(func=self._cmd_resolved_rates)

        # newton ik
        nik = sub.add_parser("newton-ik", help="Newton–Raphson pose IK")
        nik.add_argument("robot")
        nik.add_argument("--q0", required=True, help="Initial guess")
        nik.add_argument("--p", help="Target position (x,y,z)")
        nik.add_argument("--R", help="Target rotation as JSON 3x3 or path")
        nik.add_argument("--euler", default=None, help="Euler sequence name (e.g., ZYX) if --angles provided")
        nik.add_argument("--angles", help="Euler angles (deg by default) CSV/JSON")
        nik.add_argument("--deg", action="store_true", help="Interpret --angles in degrees (default)")
        nik.add_argument("--rad", dest="deg", action="store_false", help="Interpret --angles in radians")
        nik.add_argument("--max-iter", type=int, default=50)
        nik.add_argument("--tol", type=float, default=1e-8)
        nik.add_argument("--weights", help="Task weights (CSV/JSON)")
        nik.add_argument("--out")
        nik.set_defaults(func=self._cmd_newton_ik)

        # LU solve
        lus = sub.add_parser("lu-solve", help="Solve A x = b (chapter 8.5)")
        lus.add_argument("--A", required=True, help="Matrix as JSON or path")
        lus.add_argument("--b", required=True, help="Vector as JSON or path")
        lus.add_argument("--out")
        lus.set_defaults(func=self._cmd_lu_solve)

        # LU inverse
        lui = sub.add_parser("lu-inv", help="Compute A^{-1} via LU")
        lui.add_argument("--A", required=True, help="Matrix as JSON or path")
        lui.add_argument("--out")
        lui.set_defaults(func=self._cmd_lu_inv)

        # diagrams (delegated to velocity.design with safe fallback)
        dg = sub.add_parser("diagram", help="Export class diagram (optional pylint presence)")
        dg.add_argument("--out", default=self.api.default_out, help="Output dir")
        dg.set_defaults(func=self._cmd_diagram)

        return p

    # ------------------------------- entrypoint -------------------------------

    def run(self, argv: Optional[Sequence[str]] = None) -> int:
        parser = self.build_parser()
        ns = parser.parse_args(argv)
        try:
            return ns.func(ns)
        except APIError as e:
            print(f"[velocity] API error: {e}", file=sys.stderr)
            return 2
        except Exception as e:  # pragma: no cover
            print(f"[velocity] Unexpected error: {e}", file=sys.stderr)
            return 1

    # ------------------------------ subcommands ------------------------------

    def _load(self, path: str) -> RobotSpec:
        return self.api.load_robot(path)

    def _cmd_fk(self, ns: argparse.Namespace) -> int:
        spec = self._load(ns.robot)
        q = _maybe_json_or_csv(ns.q) or []
        fk = self.api.fk(spec, q)
        return _write_or_print(fk, ns.out)

    def _cmd_jacobian(self, ns: argparse.Namespace) -> int:
        spec = self._load(ns.robot)
        q = _maybe_json_or_csv(ns.q) or []
        J = self.api.jacobian_geometric(spec, q)
        return _write_or_print({"J": J}, ns.out)

    def _cmd_jacobian_analytic(self, ns: argparse.Namespace) -> int:
        spec = self._load(ns.robot)
        q = _maybe_json_or_csv(ns.q) or []
        J = self.api.jacobian_analytic(spec, q, euler=ns.euler.upper())
        return _write_or_print({"J_A": J}, ns.out)

    def _cmd_resolved_rates(self, ns: argparse.Namespace) -> int:
        spec = self._load(ns.robot)
        q = _maybe_json_or_csv(ns.q) or []
        xdot = _maybe_json_or_csv(ns.xdot) or []
        weights = _maybe_json_or_csv(ns.weights)
        qdot = self.api.resolved_rates(spec, q, xdot, damping=ns.damping, weights=weights)
        return _write_or_print({"qdot": qdot}, ns.out)

    def _cmd_newton_ik(self, ns: argparse.Namespace) -> int:
        spec = self._load(ns.robot)
        q0 = _maybe_json_or_csv(ns.q0) or []
        target: Dict[str, Any] = {}

        # Position (optional)
        if ns.p:
            target["p"] = np.asarray(_maybe_json_or_csv(ns.p), dtype=float).tolist()

        # Rotation as matrix (optional)
        if ns.R:
            R = ns.R.strip()
            if Path(R).exists():
                target["R"] = _read_json_array(R)
            else:
                target["R"] = json.loads(R)

        # Euler angles (optional)
        if ns.angles:
            angles = np.asarray(_maybe_json_or_csv(ns.angles), dtype=float)
            if ns.deg:
                angles = np.deg2rad(angles)
            if not ns.euler:
                raise APIError("Provide --euler sequence along with --angles.")
            target["euler"] = {"seq": ns.euler.upper(), "angles": angles.tolist()}

        # Decide whether to pass an Euler sequence to the API:
        # Only pass it if an orientation target is present.
        has_orientation = ("R" in target) or ("euler" in target)
        euler_seq = ns.euler.upper() if (ns.euler and has_orientation) else (target["euler"]["seq"] if "euler" in target else None)

        q_sol, info = self.api.newton_ik(
            spec,
            q0,
            target,
            max_iter=ns.max_iter,
            tol=ns.tol,
            weights=_maybe_json_or_csv(ns.weights),
            euler=euler_seq,
        )
        return _write_or_print({"q": q_sol, "info": info}, ns.out)

    def _cmd_lu_solve(self, ns: argparse.Namespace) -> int:
        A = ns.A
        if Path(A).exists():
            A = _read_json_array(A)
        else:
            A = json.loads(A)
        b = ns.b
        if Path(b).exists():
            b = _read_json_array(b)
        else:
            b = json.loads(b)
        x = self.api.lu_solve(A, b)
        return _write_or_print({"x": x}, ns.out)

    def _cmd_lu_inv(self, ns: argparse.Namespace) -> int:
        A = ns.A
        if Path(A).exists():
            A = _read_json_array(A)
        else:
            A = json.loads(A)
        inv = self.api.lu_inverse(A)
        return _write_or_print({"A_inv": inv}, ns.out)

    def _cmd_diagram(self, ns: argparse.Namespace) -> int:
        """
        Generate a class diagram without calling deprecated pylint.pyreverse APIs.

        Behavior expected by tests:
          - If `pylint.pyreverse.main` is importable on this system: success (0) and print a JSON dict.
          - Otherwise: return 2 (API/usage error).
        """
        log = logging.getLogger("velocity")
        try:
            arts = design.run_pyreverse(package_dir=Path(__file__).resolve().parent, outdir=ns.out)
            return _write_or_print(arts, None)
        except RuntimeError as e:
            log.error(str(e))
            return 2
        except Exception as e:  # pragma: no cover
            log.error("[velocity] Unexpected error: %s", e)
            return 1


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    return VelocityCLI().run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
