"""
orientation.cli (OO version)
----------------------------
Command-line interface for the Orientation Kinematics toolkit, implemented
with a Command pattern so each subcommand is a class.

File-friendly upgrades:
- Any command that consumes a 3x3 matrix now supports --matrix-file <csv>.
  Files are resolved via IOManager (prefers orientation/in, falls back to out).
- All outputs still support --save <csv> to write into orientation/out.

Design
------
- CLIContext: shared config (paths, I/O helpers) for commands.
- BaseCommand: abstract interface for commands.
- One class per verb (e.g., MatrixFromAxis, ComposeAxis, ToQuat, ...).
- OrientationCLI: orchestrates argparse, registers commands, dispatches run().
"""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Dict

import numpy as np

# Internal imports (OO types)
from .core import SO3, AxisAngle, RodriguesVector, Quaternion
from .utils import expm_so3
from .design import generate_diagram
from .io import OUT_DIR, IN_DIR, IO, write_json, write_matrix_csv


# ---------------------------------------------------------------------------
# Shared context & helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CLIContext:
    in_dir: Path = IN_DIR
    out_dir: Path = OUT_DIR

    @staticmethod
    def vec3(values: Iterable[str]) -> np.ndarray:
        v = list(map(float, values))
        if len(v) != 3:
            raise argparse.ArgumentTypeError("Expected 3 numbers for a 3-vector.")
        return np.asarray(v, dtype=float)

    @staticmethod
    def mat9(values: Iterable[str]) -> np.ndarray:
        m = list(map(float, values))
        if len(m) != 9:
            raise argparse.ArgumentTypeError("Expected 9 numbers for a 3x3 matrix (row-major).")
        return np.asarray(m, dtype=float).reshape(3, 3)

    @staticmethod
    def read_matrix_file(name: str) -> np.ndarray:
        # Uses IOManager: prefers orientation/in, falls back to orientation/out
        return IO.read_matrix_csv(name)

    def maybe_save_matrix(self, name: str | None, M: np.ndarray) -> None:
        if not name:
            return
        write_matrix_csv(name, M)
        print(f"Saved CSV -> {self.out_dir / name}")

    @staticmethod
    def print_matrix(M: np.ndarray) -> None:
        np.set_printoptions(precision=6, suppress=True)
        print(M)


# ---------------------------------------------------------------------------
# Command base
# ---------------------------------------------------------------------------

class BaseCommand(ABC):
    """Abstract CLI command. Subclasses must implement add_arguments and run."""

    name: str = ""
    help: str = ""

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None: ...

    @abstractmethod
    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None: ...

    # small utility for --save option reuse
    @staticmethod
    def add_save_option(p: argparse.ArgumentParser) -> None:
        p.add_argument("--save", type=str, help="Save 3x3 matrix as CSV in orientation/out.")


# ---------------------------------------------------------------------------
# Concrete command classes
# ---------------------------------------------------------------------------

class MatrixFromAxis(BaseCommand):
    name, help = "matrix-from-axis", "Build rotation matrix from axis-angle (φ, û)."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--axis", nargs=3, required=True, help="Unit axis vector (ux uy uz).")
        p.add_argument("--phi", type=float, required=True, help="Angle in radians.")
        self.add_save_option(p)

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        R = AxisAngle(args.phi, ctx.vec3(args.axis)).as_matrix()
        ctx.print_matrix(R)
        ctx.maybe_save_matrix(args.save, R)


class ComposeAxis(BaseCommand):
    name, help = "compose-axis", "Compose two axis-angle rotations (R2 * R1)."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--phi1", type=float, required=True)
        p.add_argument("--axis1", nargs=3, required=True)
        p.add_argument("--phi2", type=float, required=True)
        p.add_argument("--axis2", nargs=3, required=True)
        self.add_save_option(p)

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        R = (SO3.from_axis_angle(args.phi2, ctx.vec3(args.axis2)).R
             @ SO3.from_axis_angle(args.phi1, ctx.vec3(args.axis1)).R)
        ctx.print_matrix(R)
        ctx.maybe_save_matrix(args.save, R)


class ToQuat(BaseCommand):
    name, help = "to-quat", "Rotation matrix → quaternion (Euler parameters e0 e1 e2 e3)."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--matrix", nargs=9, help="3x3 rotation matrix, row-major.")
        g.add_argument("--matrix-file", type=str,
                       help="CSV filename under orientation/in (fallback: orientation/out).")

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        if args.matrix_file:
            R = ctx.read_matrix_file(args.matrix_file)
        else:
            R = ctx.mat9(args.matrix)
        q = Quaternion.from_matrix(R)
        print(f"{q.e0} {q.e1} {q.e2} {q.e3}")


class FromQuat(BaseCommand):
    name, help = "from-quat", "Quaternion (e0 e1 e2 e3) → rotation matrix."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--quat", nargs=4, required=True)
        self.add_save_option(p)

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        e0, e1, e2, e3 = map(float, args.quat)
        R = Quaternion(e0, e1, e2, e3).as_matrix()
        ctx.print_matrix(R)
        ctx.maybe_save_matrix(args.save, R)


class RodriguesToMatrix(BaseCommand):
    name, help = "rodrigues-to-matrix", "Rodrigues vector w → rotation matrix."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--w", nargs=3, required=True)
        self.add_save_option(p)

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        R = RodriguesVector(ctx.vec3(args.w)).as_matrix()
        ctx.print_matrix(R)
        ctx.maybe_save_matrix(args.save, R)


class MatrixToRodrigues(BaseCommand):
    name, help = "matrix-to-rodrigues", "Rotation matrix → Rodrigues vector w."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--matrix", nargs=9, help="3x3 rotation matrix, row-major.")
        g.add_argument("--matrix-file", type=str,
                       help="CSV filename under orientation/in (fallback: orientation/out).")

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        if args.matrix_file:
            R = ctx.read_matrix_file(args.matrix_file)
        else:
            R = ctx.mat9(args.matrix)
        aa = AxisAngle.from_matrix(R)
        w = aa.to_rodrigues().w
        print(f"{w[0]} {w[1]} {w[2]}")


class EulerToMatrix(BaseCommand):
    name, help = "euler-to-matrix", "Euler angles → rotation matrix (axis order configurable)."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--angles", nargs=3, required=True, help="Angles (default radians).")
        p.add_argument("--order", type=str, default="ZYX", help="Axis order, e.g. ZYX or XYZ.")
        p.add_argument("--deg", action="store_true", help="Interpret input angles as degrees.")
        self.add_save_option(p)

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        ang = np.asarray(list(map(float, args.angles)), dtype=float)
        if args.deg:
            ang = np.deg2rad(ang)
        axes = {"X": np.array([1.0, 0.0, 0.0]),
                "Y": np.array([0.0, 1.0, 0.0]),
                "Z": np.array([0.0, 0.0, 1.0])}
        R = np.eye(3)
        for a, k in zip(ang, args.order.upper()):
            R = SO3.from_axis_angle(a, axes[k]).R @ R
        ctx.print_matrix(R)
        ctx.maybe_save_matrix(args.save, R)


class MatrixToEuler(BaseCommand):
    name, help = "matrix-to-euler", "Rotation matrix → Euler angles (solve for given order)."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--matrix", nargs=9, help="3x3 rotation matrix, row-major.")
        g.add_argument("--matrix-file", type=str,
                       help="CSV filename under orientation/in (fallback: orientation/out).")
        p.add_argument("--order", type=str, default="ZYX")
        p.add_argument("--deg", action="store_true", help="Print angles in degrees.")

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        # Load matrix
        if args.matrix_file:
            Rtarget = ctx.read_matrix_file(args.matrix_file)
        else:
            Rtarget = ctx.mat9(args.matrix)

        order = args.order.upper()
        axes = {"X": np.array([1.0, 0.0, 0.0]),
                "Y": np.array([0.0, 1.0, 0.0]),
                "Z": np.array([0.0, 0.0, 1.0])}

        def build(a: Sequence[float]) -> np.ndarray:
            Rb = np.eye(3)
            for ang, k in zip(a, order):
                Rb = SO3.from_axis_angle(ang, axes[k]).R @ Rb
            return Rb

        x = np.zeros(3)
        for _ in range(60):
            Rb = build(x)
            r = (Rb - Rtarget).ravel()
            if np.linalg.norm(r) < 1e-12:
                break
            J = np.zeros((9, 3))
            eps = 1e-6
            for i in range(3):
                dx = x.copy(); dx[i] += eps
                J[:, i] = (build(dx) - Rb).ravel() / eps
            step, *_ = np.linalg.lstsq(J, -r, rcond=None)
            x += step
            if np.linalg.norm(step) < 1e-10:
                break

        ang = np.rad2deg(x) if args.deg else x
        print(f"{ang[0]} {ang[1]} {ang[2]}")


class ExpMap(BaseCommand):
    name, help = "expmap", "Exponential map: exp(omega^) → rotation matrix."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--omega", nargs=3, required=True, help="Rotation vector (axis*angle).")
        self.add_save_option(p)

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        R = expm_so3(ctx.vec3(args.omega))
        ctx.print_matrix(R)
        ctx.maybe_save_matrix(args.save, R)


class RandomSO3(BaseCommand):
    name, help = "random-so3", "Generate N random SO(3) rotations (uniform)."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--n", type=int, default=1)
        p.add_argument("--out", type=str, help="Write JSON to orientation/out/<out>.")

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        n = int(args.n)
        rots = []
        for _ in range(n):
            # Shoemake method (uniform on S^3 → SO(3))
            u1, u2, u3 = np.random.rand(3)
            q = Quaternion(
                np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(u1) * np.cos(2 * np.pi * u3),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),
            )
            rots.append(q.as_matrix().tolist())
        payload = {"rotations": rots}
        if args.out:
            write_json(args.out, payload)
            print(f"Wrote {n} rotations -> {ctx.out_dir / args.out}")
        else:
            print(json.dumps(payload, indent=2))


class Diagram(BaseCommand):
    name, help = "diagram", "Generate class diagrams (Graphviz .dot and Mermaid .mmd)."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", type=str, default=str(OUT_DIR),
                       help="Output directory (default orientation/out).")

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        dot, mmd = generate_diagram(Path(args.out))
        print(f"Wrote diagram files:\n - {dot}\n - {mmd}")


class Batch(BaseCommand):
    name, help = "batch", "Run a JSON batch of operations from orientation/in."

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--in", dest="infile", required=True, help="Input JSON in orientation/in/")
        p.add_argument("--out", dest="outfile", required=True, help="Output JSON in orientation/out/")

    def run(self, args: argparse.Namespace, ctx: CLIContext) -> None:
        jobs = json.loads((ctx.in_dir / args.infile).read_text())
        results: List[Dict] = []
        # Thin dispatcher using OO classes
        for job in jobs:
            op = job.get("op", "")
            p = job.get("params", {}) or {}
            try:
                if op == MatrixFromAxis.name:
                    R = AxisAngle(p["phi"], np.array(p["axis"], dtype=float)).as_matrix()
                    results.append({"op": op, "result": R.tolist()})
                elif op == ComposeAxis.name:
                    R = (SO3.from_axis_angle(p["phi2"], np.array(p["axis2"], dtype=float)).R
                         @ SO3.from_axis_angle(p["phi1"], np.array(p["axis1"], dtype=float)).R)
                    results.append({"op": op, "result": R.tolist()})
                elif op == ToQuat.name:
                    R = np.asarray(p["matrix"], dtype=float).reshape(3, 3)
                    q = Quaternion.from_matrix(R)
                    results.append({"op": op, "result": [q.e0, q.e1, q.e2, q.e3]})
                elif op == FromQuat.name:
                    R = Quaternion(*p["quat"]).as_matrix()
                    results.append({"op": op, "result": np.asarray(R).tolist()})
                elif op == RodriguesToMatrix.name:
                    R = RodriguesVector(np.asarray(p["w"], dtype=float)).as_matrix()
                    results.append({"op": op, "result": np.asarray(R).tolist()})
                elif op == MatrixToRodrigues.name:
                    R = np.asarray(p["matrix"], dtype=float).reshape(3, 3)
                    aa = AxisAngle.from_matrix(R)
                    results.append({"op": op, "result": aa.to_rodrigues().w.tolist()})
                elif op == EulerToMatrix.name:
                    ang = np.asarray(p["angles"], dtype=float)
                    if p.get("deg", False):
                        ang = np.deg2rad(ang)
                    order = p.get("order", "ZYX").upper()
                    axes = {"X": np.array([1.0, 0.0, 0.0]),
                            "Y": np.array([0.0, 1.0, 0.0]),
                            "Z": np.array([0.0, 0.0, 1.0])}
                    R = np.eye(3)
                    for a, k in zip(ang, order):
                        R = SO3.from_axis_angle(a, axes[k]).R @ R
                    results.append({"op": op, "result": R.tolist()})
                elif op == ExpMap.name:
                    R = expm_so3(np.asarray(p["omega"], dtype=float))
                    results.append({"op": op, "result": np.asarray(R).tolist()})
                else:
                    results.append({"op": op, "error": f"unknown op '{op}'"})
            except Exception as exc:
                results.append({"op": op, "error": f"{type(exc).__name__}: {exc}"})

        write_json(args.outfile, {"results": results})
        print(f"Wrote batch results -> {ctx.out_dir / args.outfile}")


# ---------------------------------------------------------------------------
# CLI Orchestrator
# ---------------------------------------------------------------------------

class OrientationCLI:
    """Builds the argparse tree and dispatches to command classes."""

    def __init__(self, ctx: CLIContext | None = None):
        self.ctx = ctx or CLIContext()
        self.parser = argparse.ArgumentParser(
            prog="orientation",
            description="Orientation Kinematics CLI (OO)"
        )
        self.subparsers = self.parser.add_subparsers(dest="cmd", required=True)
        self._commands: Dict[str, BaseCommand] = {}
        self._register_default_commands()

    def _register(self, cmd: BaseCommand) -> None:
        sp = self.subparsers.add_parser(cmd.name, help=cmd.help)
        cmd.add_arguments(sp)
        sp.set_defaults(_cmd=cmd)
        self._commands[cmd.name] = cmd

    def _register_default_commands(self) -> None:
        for cls in (
            MatrixFromAxis, ComposeAxis, ToQuat, FromQuat,
            RodriguesToMatrix, MatrixToRodrigues,
            EulerToMatrix, MatrixToEuler,
            ExpMap, RandomSO3, Diagram, Batch
        ):
            self._register(cls())

    def run(self, argv: List[str] | None = None) -> None:
        args = self.parser.parse_args(argv)
        cmd: BaseCommand = getattr(args, "_cmd")
        cmd.run(args, self.ctx)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    OrientationCLI().run(argv)


if __name__ == "__main__":
    main()
