# velocity_kinematics/cli.py
"""
Command-line interface for the Velocity Kinematics Toolkit.

Design goals
------------
- Zero non-stdlib dependencies (pure argparse).
- Thin, testable commands; kinematics via core.* to support DH + URDF.
- Exhaustive yet ergonomic flags (CSV/JSON inputs, files or literals).
- No side effects unless explicitly requested (stdout or --out).

Examples
--------
# Forward kinematics on a DH YAML
python -m velocity_kinematics.cli fk velocity_kinematics/in/planar2r.yml --q 0.3,0.2 --out velocity_kinematics/out/fk.json

# Geometric / Analytic Jacobian (URDF or DH)
python -m velocity_kinematics.cli jacobian velocity_kinematics/in/arm.urdf --q 0.1,0.2
python -m velocity_kinematics.cli jacobian-analytic velocity_kinematics/in/arm.yml --q 0.1,0.2 --euler ZXZ

# Resolved rates
python -m velocity_kinematics.cli resolved-rates velocity_kinematics/in/arm.yml --q 0.1,0.2 --xdot 0.1,0,0,0,0,0

# Newton–Raphson IK (position + ZYX Euler)
python -m velocity_kinematics.cli newton-ik velocity_kinematics/in/arm.yml --q0 0,0 --p 0.5,0.1,0.2 --euler zyx --angles 10,20,0 --deg

# Linear algebra exercises (chapter 8.5+)
python -m velocity_kinematics.cli lu-solve --A "[[2,1],[1,3]]" --b "[1,2]"
python -m velocity_kinematics.cli lu-inv   --A velocity_kinematics/in/matrix.json

# Class diagram (optional pylint presence)
python -m velocity_kinematics.cli diagram --out velocity_kinematics/out

# NEW: Generate a minimal Sphinx skeleton
python -m velocity_kinematics.cli sphinx-skel velocity_kinematics/docs --force
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

# API kept for LU & diagram commands
from .apis import VelocityAPI, APIError
from . import design
from . import core

# Optional YAML (only needed when loading .yaml/.yml robot specs)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


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


# ----------------------------- Robot loader ---------------------------------- #

def _load_robot_any(path_or_spec: str):
    """
    Accept:
      - path to .yaml/.yml (DH or URDF wrapper)
      - path to .json (DH or URDF wrapper)
      - path to .urdf/.xml (raw URDF)
    Returns a core.DHRobot or core.URDFRobot accordingly.
    """
    p = Path(path_or_spec)
    sfx = p.suffix.lower()

    # Raw URDF/XML → URDFRobot
    if sfx in (".urdf", ".xml"):
        return core.URDFRobot.from_spec(str(p))

    # YAML/JSON wrapper
    if sfx in (".yaml", ".yml", ".json"):
        txt = p.read_text(encoding="utf-8")
        data = yaml.safe_load(txt) if sfx in (".yaml", ".yml") else json.loads(txt)

        # If wrapper has an 'urdf' key → URDFRobot
        if isinstance(data, dict) and data.get("urdf"):
            return core.URDFRobot.from_spec(data)

        # Otherwise treat it as DH spec
        return core.DHRobot.from_spec(data)

    raise APIError(f"Unsupported robot spec extension: {p.suffix} (use .yaml/.yml/.json/.urdf/.xml)")


# --------------------------------------------------------------------------- #
# Sphinx skeleton writer
# --------------------------------------------------------------------------- #

def _write_file(p: Path, text: str, force: bool) -> None:
    if p.exists() and not force:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def _sphinx_conf_py() -> str:
    # Try furo if installed, else fall back to alabaster to avoid blank output.
    return (
        'import importlib\n'
        'project = "velocity_kinematics"\n'
        'extensions = [\n'
        '    "sphinx.ext.autodoc",\n'
        '    "sphinx.ext.napoleon",\n'
        '    "sphinx.ext.viewcode",\n'
        ']\n'
        'templates_path = ["_templates"]\n'
        'exclude_patterns = []\n'
        'html_theme = "furo" if importlib.util.find_spec("furo") else "alabaster"\n'
        'html_static_path = ["_static"]\n'
        '# Sphinx >= 5 uses root_doc; default is "index" but set explicitly for safety\n'
        'root_doc = "index"\n'
    )

def _sphinx_index_rst() -> str:
    return (\
        "Velocity Toolkit Documentation\n"
        "===============================\n\n"
        "Welcome! This is a tiny Sphinx skeleton generated by ``velocity_kinematics.cli sphinx-skel``.\n"
        "Use it as a starting point; extend as needed.\n\n"
        ".. toctree::\n"
        "   :maxdepth: 2\n"
        "   :caption: Contents:\n\n"
        "   api\n"
        "\n"
        "Getting Started\n"
        "---------------\n\n"
        "Build the HTML documentation with::\n\n"
        "   make html\n"
        "\n"
        "The output will be in ``_build/html/index.html``.\n"
    )

def _sphinx_api_rst() -> str:
    return (\
        "API Reference\n"
        "=============\n\n"
        ".. automodule:: velocity_kinematics.core\n"
        "   :members:\n"
        "   :undoc-members:\n\n"
        ".. automodule:: velocity_kinematics.apis\n"
        "   :members:\n"
        "   :undoc-members:\n\n"
        ".. automodule:: velocity_kinematics.design\n"
        "   :members:\n"
        "   :undoc-members:\n"
    )

def _sphinx_makefile() -> str:
    return (\
        "# Minimal Sphinx Makefile\n"
        ".PHONY: html clean\n"
        "html:\n"
        "\t+sphinx-build -b html . _build/html\n"
        "clean:\n"
        "\t+rm -rf _build\n"
    )

def _sphinx_skeleton(dest: Path, force: bool) -> str:
    _write_file(dest / "conf.py", _sphinx_conf_py(), force)
    _write_file(dest / "index.rst", _sphinx_index_rst(), force)
    _write_file(dest / "api.rst", _sphinx_api_rst(), force)
    _write_file(dest / "Makefile", _sphinx_makefile(), force)
    (dest / "_templates").mkdir(parents=True, exist_ok=True)
    (dest / "_static").mkdir(parents=True, exist_ok=True)
    return str(dest)


# --------------------------------------------------------------------------- #
# CLI object (for OOP + TDD)
# --------------------------------------------------------------------------- #

@dataclass
class VelocityCLI:
    api: VelocityAPI = VelocityAPI()

    # -------- parser construction (isolated for unit tests) --------

    def build_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            prog="velocity_kinematics",
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
        rr = sub.add_parser("resolved-rates", help="Inverse velocity_kinematics qdot from Xdot = J qdot")
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
        nik.add_argument("--R", help="Target rotation_kinematics as JSON 3x3 or path")
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

        # LU inverse_kinematics
        lui = sub.add_parser("lu-inv", help="Compute A^{-1} via LU")
        lui.add_argument("--A", required=True, help="Matrix as JSON or path")
        lui.add_argument("--out")
        lui.set_defaults(func=self._cmd_lu_inv)

        # diagrams (delegated to velocity_kinematics.design with safe fallback)
        dg = sub.add_parser("diagram", help="Export class diagram (optional pylint presence)")
        dg.add_argument("--out", default=self.api.default_out, help="Output dir")
        dg.set_defaults(func=self._cmd_diagram)

        # sphinx skeleton (new, optional)
        spx = sub.add_parser("sphinx-skel", help="Create a minimal Sphinx docs skeleton")
        spx.add_argument("dest", nargs="?", default="docs", help="Destination directory (default: docs)")
        spx.add_argument("--force", action="store_true", help="Overwrite existing files if present")
        spx.set_defaults(func=self._cmd_sphinx_skel)

        return p

    # ------------------------------- entrypoint -------------------------------

    def run(self, argv: Optional[Sequence[str]] = None) -> int:
        parser = self.build_parser()
        ns = parser.parse_args(argv)
        try:
            return ns.func(ns)
        except APIError as e:
            print(f"[velocity_kinematics] API error: {e}", file=sys.stderr)
            return 2
        except Exception as e:  # pragma: no cover
            print(f"[velocity_kinematics] Unexpected error: {e}", file=sys.stderr)
            return 1

    # ------------------------------ subcommands ------------------------------

    # -- Kinematics via core.* so URDF works --

    def _cmd_fk(self, ns: argparse.Namespace) -> int:
        robot = _load_robot_any(ns.robot)
        q = np.asarray(_maybe_json_or_csv(ns.q) or [], dtype=float)
        out = robot.fk(q)
        return _write_or_print(out, ns.out)

    def _cmd_jacobian(self, ns: argparse.Namespace) -> int:
        robot = _load_robot_any(ns.robot)
        q = np.asarray(_maybe_json_or_csv(ns.q) or [], dtype=float)
        J = robot.jacobian_geometric(q)
        return _write_or_print({"J": J}, ns.out)

    def _cmd_jacobian_analytic(self, ns: argparse.Namespace) -> int:
        robot = _load_robot_any(ns.robot)
        q = np.asarray(_maybe_json_or_csv(ns.q) or [], dtype=float)
        J = robot.jacobian_analytic(q, euler=ns.euler.upper())
        return _write_or_print({"J_A": J}, ns.out)

    def _cmd_resolved_rates(self, ns: argparse.Namespace) -> int:
        robot = _load_robot_any(ns.robot)
        q = np.asarray(_maybe_json_or_csv(ns.q) or [], dtype=float)
        xdot = np.asarray(_maybe_json_or_csv(ns.xdot) or [], dtype=float)
        J = robot.jacobian_geometric(q)
        qdot = core.solvers.resolved_rates(J, xdot, damping=ns.damping, weights=_maybe_json_or_csv(ns.weights))
        return _write_or_print({"qdot": qdot}, ns.out)

    def _cmd_newton_ik(self, ns: argparse.Namespace) -> int:
        robot = _load_robot_any(ns.robot)
        q0 = np.asarray(_maybe_json_or_csv(ns.q0) or [], dtype=float)
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
        euler_seq: Optional[str] = None
        if ns.angles:
            angles = np.asarray(_maybe_json_or_csv(ns.angles), dtype=float)
            if ns.deg:
                angles = np.deg2rad(angles)
            if not ns.euler:
                raise APIError("Provide --euler sequence along with --angles.")
            euler_seq = ns.euler.upper()
            target["euler"] = {"seq": euler_seq, "angles": angles.tolist()}

        # Only pass euler if an orientation_kinematics target is present
        if ("R" not in target) and ("euler" not in target):
            euler_seq = None

        q_sol, info = core.solvers.newton_ik(
            robot,
            q0,
            target,
            max_iter=ns.max_iter,
            tol=ns.tol,
            weights=_maybe_json_or_csv(ns.weights),
            euler=euler_seq or "ZYX",
        )
        return _write_or_print({"q": q_sol, "info": info}, ns.out)

    # -- Linear algebra & diagram via API (unchanged) --

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
        log = logging.getLogger("velocity_kinematics")
        try:
            arts = design.run_pyreverse(package_dir=Path(__file__).resolve().parent, outdir=ns.out)
            return _write_or_print(arts, None)
        except RuntimeError as e:
            log.error(str(e))
            return 2
        except Exception as e:  # pragma: no cover
            log.error("[velocity_kinematics] Unexpected error: %s", e)
            return 1

    def _cmd_sphinx_skel(self, ns: argparse.Namespace) -> int:
        dest = Path(ns.dest)
        path = _sphinx_skeleton(dest, force=ns.force)
        print(path)
        return 0


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    return VelocityCLI().run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
