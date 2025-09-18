"""
orientation.apis (OO version)
-----------------------------
High-level, object-oriented facade for orientation kinematics operations.

Design
------
- DecompositionSolver (abstract): strategy for solving R ≈ Rc(γ) Rb(β) Ra(α).
- SciPyDecompositionSolver: uses scipy.optimize.least_squares (if available).
- NumpyGNDecompositionSolver: pure-NumPy Gauss-Newton fallback (deterministic).
- OrientationService: the main facade object exposing user-facing operations:
    * composition & conversions among AxisAngle / Rodrigues / Quaternion / SO3
    * Euler angles in/out for arbitrary axis order
    * exponential map exp(omega^) → SO(3)
    * uniform random SO(3) sampling (Shoemake)
    * decomposition into three arbitrary axes
    * simple batch job runner + file-based batch using IOManager

Everything is dependency-injectable: utils (math backend), io manager, and
the solver strategy can be swapped in tests.

References
----------
- AxisAngle, Rodrigues, Quaternion, SO3 methods match book eqs. 3.1–3.434
- Exponential map: exp(ω^) (Eq. 3.187)
- Cayley / Rodrigues vector: w = û tan(φ/2) (Eqs. 3.201–3.206)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .core import SO3, AxisAngle, RodriguesVector, Quaternion
from .utils import OrientationUtils, UTILS
from .io import IOManager, IO


# --------------------------------------------------------------------------
# Decomposition solvers (strategy)
# --------------------------------------------------------------------------

class DecompositionSolver(Protocol):
    """Strategy interface: solve R ≈ Rc(γ) Rb(β) Ra(α)."""

    def solve(
        self,
        R_target: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x0: Optional[np.ndarray] = None,
        max_iters: int = 60,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Return angles [α, β, γ] that minimize ||Rc Rb Ra - R_target||_F."""


@dataclass
class SciPyDecompositionSolver:
    """SciPy-based non-linear least squares (if SciPy present)."""
    def solve(
        self,
        R_target: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x0: Optional[np.ndarray] = None,
        max_iters: int = 60,
        tol: float = 1e-10,
    ) -> np.ndarray:
        try:
            from scipy.optimize import least_squares  # type: ignore
        except Exception as exc:  # pragma: no cover (guard)
            raise RuntimeError("SciPy not available") from exc

        R_target = np.asarray(R_target, dtype=float).reshape(3, 3)
        a = np.asarray(a, dtype=float).reshape(3)
        b = np.asarray(b, dtype=float).reshape(3)
        c = np.asarray(c, dtype=float).reshape(3)

        def make_R(x: np.ndarray) -> np.ndarray:
            α, β, γ = x
            Ra = SO3.from_axis_angle(α, a).R
            Rb = SO3.from_axis_angle(β, b).R
            Rc = SO3.from_axis_angle(γ, c).R
            return Rc @ Rb @ Ra

        def resid(x: np.ndarray) -> np.ndarray:
            return (make_R(x) - R_target).ravel()

        x0_ = np.zeros(3) if x0 is None else np.asarray(x0, dtype=float).reshape(3)
        res = least_squares(resid, x0_, max_nfev=max_iters, xtol=tol, ftol=tol, gtol=tol)
        return res.x


@dataclass
class NumpyGNDecompositionSolver:
    """Pure-NumPy Gauss-Newton least squares (deterministic)."""

    def solve(
        self,
        R_target: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x0: Optional[np.ndarray] = None,
        max_iters: int = 60,
        tol: float = 1e-10,
    ) -> np.ndarray:
        R_target = np.asarray(R_target, dtype=float).reshape(3, 3)
        a = np.asarray(a, dtype=float).reshape(3)
        b = np.asarray(b, dtype=float).reshape(3)
        c = np.asarray(c, dtype=float).reshape(3)

        def make_R(x: np.ndarray) -> np.ndarray:
            α, β, γ = x
            Ra = SO3.from_axis_angle(α, a).R
            Rb = SO3.from_axis_angle(β, b).R
            Rc = SO3.from_axis_angle(γ, c).R
            return Rc @ Rb @ Ra

        def resid(x: np.ndarray) -> np.ndarray:
            return (make_R(x) - R_target).ravel()

        x = np.zeros(3) if x0 is None else np.asarray(x0, dtype=float).reshape(3)
        for _ in range(max_iters):
            Rb = make_R(x)
            r = (Rb - R_target).ravel()
            if np.linalg.norm(r) < tol:
                break
            J = np.zeros((9, 3))
            h = 1e-6
            for i in range(3):
                dx = x.copy(); dx[i] += h
                J[:, i] = (make_R(dx) - Rb).ravel() / h
            step, *_ = np.linalg.lstsq(J, -r, rcond=None)
            x = x + step
            if np.linalg.norm(step) < tol:
                break
        return x


# --------------------------------------------------------------------------
# High-level OO facade
# --------------------------------------------------------------------------

@dataclass
class OrientationService:
    """High-level facade for orientation operations.

    Parameters
    ----------
    utils : OrientationUtils
        Math utility backend (configurable; default uses module singleton).
    io : IOManager
        IO manager for reading/writing batch jobs (default module singleton).
    solver : DecompositionSolver
        Strategy for angle decomposition. If None, tries SciPy then NumPy.
    """
    utils: OrientationUtils = UTILS
    io: IOManager = IO
    solver: Optional[DecompositionSolver] = None

    # --- internal ---

    def _solver(self) -> DecompositionSolver:
        if self.solver is not None:
            return self.solver
        # Pick SciPy if available; otherwise NumPy fallback
        try:
            import scipy  # type: ignore
            return SciPyDecompositionSolver()
        except Exception:
            return NumpyGNDecompositionSolver()

    # --- composition & conversions ---

    def compose_axis_angle(
        self,
        phi1: float,
        u1: Iterable[float],
        phi2: float,
        u2: Iterable[float],
    ) -> SO3:
        """Return SO3 for R2 * R1 from two axis-angle rotations."""
        R = (SO3.from_axis_angle(phi2, np.asarray(u2, dtype=float)).R
             @ SO3.from_axis_angle(phi1, np.asarray(u1, dtype=float)).R)
        return SO3(R)

    def matrix_to_quaternion(self, R: np.ndarray) -> Quaternion:
        return Quaternion.from_matrix(R)

    def quaternion_to_matrix(self, q: Quaternion | Sequence[float]) -> np.ndarray:
        qq = q if isinstance(q, Quaternion) else Quaternion(*q)
        return qq.as_matrix()

    def rodrigues_to_matrix(self, w: Iterable[float]) -> np.ndarray:
        return RodriguesVector(np.asarray(w, dtype=float)).as_matrix()

    def matrix_to_rodrigues(self, R: np.ndarray) -> np.ndarray:
        aa = AxisAngle.from_matrix(np.asarray(R, dtype=float))
        return aa.to_rodrigues().w

    # --- Euler angles I/O (arbitrary order) ---

    def euler_to_matrix(
        self,
        angles: Iterable[float],
        order: str = "ZYX",
        degrees: bool = False,
    ) -> np.ndarray:
        ang = np.asarray(list(angles), dtype=float)
        if degrees:
            ang = np.deg2rad(ang)
        axes = {"X": np.array([1.0, 0.0, 0.0]),
                "Y": np.array([0.0, 1.0, 0.0]),
                "Z": np.array([0.0, 0.0, 1.0])}
        R = np.eye(3)
        for a, k in zip(ang, order.upper()):
            R = SO3.from_axis_angle(a, axes[k]).R @ R
        return R

    def matrix_to_euler(
        self,
        R: np.ndarray,
        order: str = "ZYX",
        degrees: bool = False,
    ) -> np.ndarray:
        """Numerical solve of angles for given axis order."""
        R = np.asarray(R, dtype=float).reshape(3, 3)
        axes = {"X": np.array([1.0, 0.0, 0.0]),
                "Y": np.array([0.0, 1.0, 0.0]),
                "Z": np.array([0.0, 0.0, 1.0])}

        def make_R(x: np.ndarray) -> np.ndarray:
            Rb = np.eye(3)
            for a, k in zip(x, order.upper()):
                Rb = SO3.from_axis_angle(a, axes[k]).R @ Rb
            return Rb

        # Solve with the same GN routine; reuse the NumpyGNDecompositionSolver core
        solver = NumpyGNDecompositionSolver()

        def residual(x: np.ndarray) -> np.ndarray:
            return (make_R(x) - R).ravel()

        x = np.zeros(3)
        x = solver.solve(R, axes[order[0].upper()], axes[order[1].upper()], axes[order[2].upper()], x0=x)
        if degrees:
            x = np.rad2deg(x)
        return x

    # --- Exponential map ---

    def expmap(self, omega: Iterable[float]) -> np.ndarray:
        """Compute exp(omega^) using utils backend (SciPy or closed form)."""
        return self.utils.expm_so3(np.asarray(omega, dtype=float))

    # --- Random SO(3) sampling (Shoemake) ---

    def random_so3(self, n: int = 1) -> List[np.ndarray]:
        rots: List[np.ndarray] = []
        for _ in range(int(n)):
            u1, u2, u3 = np.random.rand(3)
            q = Quaternion(
                np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(u1) * np.cos(2 * np.pi * u3),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),
            )
            rots.append(q.as_matrix())
        return rots

    # --- Decomposition into three axes ---

    def decompose_into_axes(
        self,
        R: np.ndarray,
        a: Iterable[float],
        b: Iterable[float],
        c: Iterable[float],
        x0: Optional[Iterable[float]] = None,
    ) -> Tuple[float, float, float]:
        """Solve angles [α, β, γ] s.t. R ≈ Rc(γ) Rb(β) Ra(α)."""
        solver = self._solver()
        x = solver.solve(
            R_target=np.asarray(R, dtype=float),
            a=np.asarray(a, dtype=float),
            b=np.asarray(b, dtype=float),
            c=np.asarray(c, dtype=float),
            x0=None if x0 is None else np.asarray(x0, dtype=float),
        )
        return float(x[0]), float(x[1]), float(x[2])

    # --- Batch processing ---

    def run_jobs(self, jobs: List[dict]) -> List[dict]:
        """Run a list of job dicts: {'op': <name>, 'params': {...}}."""
        out: List[dict] = []
        for j in jobs:
            op = j.get("op", "")
            p = j.get("params", {}) or {}
            try:
                if op == "matrix-from-axis":
                    R = AxisAngle(p["phi"], np.array(p["axis"])).as_matrix()
                    out.append({"op": op, "result": R.tolist()})
                elif op == "compose-axis":
                    R = self.compose_axis_angle(p["phi1"], p["axis1"], p["phi2"], p["axis2"]).R
                    out.append({"op": op, "result": R.tolist()})
                elif op == "to-quat":
                    q = self.matrix_to_quaternion(np.array(p["matrix"]).reshape(3, 3))
                    out.append({"op": op, "result": [q.e0, q.e1, q.e2, q.e3]})
                elif op == "from-quat":
                    R = self.quaternion_to_matrix(p["quat"])
                    out.append({"op": op, "result": np.asarray(R).tolist()})
                elif op == "rodrigues-to-matrix":
                    R = self.rodrigues_to_matrix(p["w"])
                    out.append({"op": op, "result": np.asarray(R).tolist()})
                elif op == "matrix-to-rodrigues":
                    w = self.matrix_to_rodrigues(np.array(p["matrix"]).reshape(3, 3))
                    out.append({"op": op, "result": np.asarray(w).tolist()})
                elif op == "euler-to-matrix":
                    R = self.euler_to_matrix(
                        p["angles"],
                        order=p.get("order", "ZYX"),
                        degrees=bool(p.get("deg", False)),
                    )
                    out.append({"op": op, "result": np.asarray(R).tolist()})
                elif op == "matrix-to-euler":
                    ang = self.matrix_to_euler(
                        np.array(p["matrix"]).reshape(3, 3),
                        order=p.get("order", "ZYX"),
                        degrees=bool(p.get("deg", False)),
                    )
                    out.append({"op": op, "result": np.asarray(ang).tolist()})
                elif op == "expmap":
                    R = self.expmap(p["omega"])
                    out.append({"op": op, "result": np.asarray(R).tolist()})
                elif op == "random-so3":
                    mats = self.random_so3(int(p.get("n", 1)))
                    out.append({"op": op, "result": [m.tolist() for m in mats]})
                elif op == "decompose-axes":
                    α, β, γ = self.decompose_into_axes(
                        p["R"], p["a"], p["b"], p["c"], x0=p.get("x0")
                    )
                    out.append({"op": op, "result": [α, β, γ]})
                else:
                    out.append({"op": op, "error": f"unknown op '{op}'"})
            # Keep batch resilient
            except Exception as exc:
                out.append({"op": op, "error": f"{type(exc).__name__}: {exc}"})
        return out

    def run_jobs_from_files(self, infile: str, outfile: str) -> None:
        """Load jobs from `in/` and write results to `out/` via IO manager."""
        jobs = self.io.read_jobs(infile)
        results = self.run_jobs(jobs)
        self.io.write_results(outfile, results)


# --------------------------------------------------------------------------
# Convenience: default service singleton
# --------------------------------------------------------------------------

# Default service instance (easy to import in notebooks or scripts)
SERVICE = OrientationService()
