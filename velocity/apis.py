# velocity/apis.py
"""
Public API façade for the Velocity Kinematics Toolkit.

This module exposes a small, well-documented surface that higher layers
(CLI, notebooks, apps) can call without worrying about low-level details.

Design goals
------------
- Pure functions / small classes with deterministic behavior for easy testing.
- Lazy imports to avoid circular dependencies during piecemeal development.
- Rich type hints and friendly exceptions for better UX.
- No I/O side effects (except optional class-diagram export).

Example
-------
>>> from velocity.apis import VelocityAPI
>>> api = VelocityAPI()
>>> spec = api.load_robot("velocity/in/planar2r.yml")   # DH or URDF
>>> J = api.jacobian_geometric(spec, q=[0.3, 0.2])
>>> qdot = api.resolved_rates(spec, q=[0.3, 0.2], xdot=[0.1, 0.0, 0, 0, 0, 0])

"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "APIError",
    "RobotFormat",
    "RobotSpec",
    "VelocityAPI",
]

NDArray = np.ndarray


# --------------------------------------------------------------------------- #
# Utilities & decorators
# --------------------------------------------------------------------------- #

class APIError(RuntimeError):
    """Top-level exception for public API failures."""


def _timeit(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log wall time for API calls (visible with DEBUG level)."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        log = logging.getLogger(f"velocity.api.{fn.__name__}")
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            log.debug("done in %.2f ms", dt)
    return wrapper


# --------------------------------------------------------------------------- #
# Robot specification (format-agnostic shell used across the API)
# --------------------------------------------------------------------------- #

class RobotFormat(str, Enum):
    DH = "dh"
    URDF = "urdf"


@dataclass(frozen=True)
class RobotSpec:
    """
    Format-agnostic robot specification passed around the API.

    Attributes
    ----------
    fmt : RobotFormat
        Source format (DH or URDF).
    data : Mapping[str, Any]
        Parsed content (DH dict or URDF metadata).
    path : Optional[Path]
        Where it was loaded from (if applicable).
    name : Optional[str]
        Human-friendly robot name.
    """
    fmt: RobotFormat
    data: Mapping[str, Any]
    path: Optional[Path] = None
    name: Optional[str] = None


# --------------------------------------------------------------------------- #
# Public façade
# --------------------------------------------------------------------------- #

class VelocityAPI:
    """
    High-level façade over core functionality (FK, Jacobians, IK, linear algebra).

    Notes
    -----
    - All methods are thin wrappers that import the underlying implementations
      lazily from sibling modules to keep this file standalone in early phases.
    - Methods accept and return NumPy arrays for numeric work.
    - Each call is decorated with `_timeit` for lightweight timing logs.
    """

    def __init__(self, default_in: Union[str, Path] = "velocity/in", default_out: Union[str, Path] = "velocity/out"):
        self.default_in = Path(default_in)
        self.default_out = Path(default_out)
        self.log = logging.getLogger(self.__class__.__name__)

    # ------------------------------- I/O & Loading -------------------------------

    @_timeit
    def load_robot(self, source: Union[str, Path]) -> RobotSpec:
        """
        Load a robot description from YAML/JSON (DH) or URDF file.

        Parameters
        ----------
        source : str or Path
            Path to file. Extension decides the loader:
            - .yml/.yaml/.json -> DH dictionary with conventional keys
            - .urdf/.xml      -> URDF model (parsed to a dict-like structure)

        Returns
        -------
        RobotSpec
        """
        p = Path(source)
        try:
            # Lazy import to avoid circulars during piecemeal development
            from .io import load_dh_from_file, load_urdf_from_file  # type: ignore
        except Exception as e:  # pragma: no cover
            raise APIError(f"Required I/O functions not available yet: {e}") from e

        suffix = p.suffix.lower()
        if suffix in {".yml", ".yaml", ".json"}:
            data = load_dh_from_file(p)
            return RobotSpec(fmt=RobotFormat.DH, data=data, path=p, name=data.get("name"))
        elif suffix in {".urdf", ".xml"}:
            data = load_urdf_from_file(p)
            name = data.get("name") or p.stem
            return RobotSpec(fmt=RobotFormat.URDF, data=data, path=p, name=name)
        else:
            raise APIError(f"Unsupported robot file type: {p.suffix}")

    # --------------------------------- Kinematics --------------------------------

    @_timeit
    def fk(self, spec: RobotSpec, q: Sequence[float]) -> Mapping[str, NDArray]:
        """
        Forward kinematics: returns end-effector pose and selected intermediate frames.

        Returns
        -------
        dict
            Contains at least:
              - 'T_0e' : 4x4 homogeneous transform base->EE
              - optionally per-link transforms (implementation dependent)
        """
        try:
            from .core import DHRobot, URDFRobot  # type: ignore
        except Exception as e:  # pragma: no cover
            raise APIError(f"Core kinematics not available yet: {e}") from e

        robot = DHRobot.from_spec(spec) if spec.fmt is RobotFormat.DH else URDFRobot.from_spec(spec)
        return robot.fk(np.asarray(q, dtype=float))

    @_timeit
    def jacobian_geometric(self, spec: RobotSpec, q: Sequence[float]) -> NDArray:
        """
        Geometric Jacobian (maps joint speeds to [v; ω]).

        Returns
        -------
        np.ndarray shape (6, n)
        """
        try:
            from .core import DHRobot, URDFRobot  # type: ignore
        except Exception as e:  # pragma: no cover
            raise APIError(f"Core kinematics not available yet: {e}") from e

        robot = DHRobot.from_spec(spec) if spec.fmt is RobotFormat.DH else URDFRobot.from_spec(spec)
        return robot.jacobian_geometric(np.asarray(q, dtype=float))

    @_timeit
    def jacobian_analytic(
        self,
        spec: RobotSpec,
        q: Sequence[float],
        euler: str = "ZYX",
    ) -> NDArray:
        """
        Analytic Jacobian J_A (translational + Euler rate mapping).

        Parameters
        ----------
        euler : str
            Euler sequence for analytic mapping (e.g., 'ZYX', 'ZXZ').
        """
        try:
            from .core import DHRobot, URDFRobot  # type: ignore
        except Exception as e:  # pragma: no cover
            raise APIError(f"Core kinematics not available yet: {e}") from e

        robot = DHRobot.from_spec(spec) if spec.fmt is RobotFormat.DH else URDFRobot.from_spec(spec)
        return robot.jacobian_analytic(np.asarray(q, dtype=float), euler=euler)

    @_timeit
    def resolved_rates(
        self,
        spec: RobotSpec,
        q: Sequence[float],
        xdot: Sequence[float],
        damping: float | None = None,
        weights: Optional[Sequence[float]] = None,
    ) -> NDArray:
        """
        Solve inverse velocity (resolved-rates) q̇ from Ẋ = J q̇.

        If the system is square and nonsingular: J^{-1} Ẋ.
        Otherwise uses (weighted) damped least squares.

        Parameters
        ----------
        damping : float, optional
            Tikhonov damping (λ). If None, an adaptive λ may be used by core.
        weights : sequence of floats, optional
            Diagonal weights for task space residuals.

        Returns
        -------
        np.ndarray (n,)
        """
        try:
            from .core import solvers  # type: ignore
        except Exception as e:  # pragma: no cover
            raise APIError(f"Solver backends not available yet: {e}") from e

        J = self.jacobian_geometric(spec, q)
        xdot = np.asarray(xdot, dtype=float).reshape(-1)
        return solvers.resolved_rates(J, xdot, damping=damping, weights=weights)

    @_timeit
    def newton_ik(
        self,
        spec: RobotSpec,
        q0: Sequence[float],
        x_target: Mapping[str, Any],
        max_iter: int = 50,
        tol: float = 1e-8,
        weights: Optional[Sequence[float]] = None,
        euler: str = "ZYX",
    ) -> Tuple[NDArray, Dict[str, Any]]:
        """
        Newton–Raphson inverse kinematics on pose (position + orientation).

        Parameters
        ----------
        x_target : Mapping[str, Any]
            Expected keys: 'p' (3,), and either 'R' (3x3) or 'euler' (sequence).
        weights : optional task weights for least squares step.
        euler : Euler sequence used when target orientation is given as Euler angles.

        Returns
        -------
        (q_sol, info) : (np.ndarray, dict)
            info contains diagnostics (iters, residual norms, converged, etc.)
        """
        try:
            from .core import DHRobot, URDFRobot, solvers  # type: ignore
        except Exception as e:  # pragma: no cover
            raise APIError(f"IK backends not available yet: {e}") from e

        robot = DHRobot.from_spec(spec) if spec.fmt is RobotFormat.DH else URDFRobot.from_spec(spec)
        return solvers.newton_ik(robot, np.asarray(q0, dtype=float), x_target, max_iter=max_iter, tol=tol, weights=weights, euler=euler)

    # -------------------------------- Linear Algebra -----------------------------

    @_timeit
    def lu_solve(self, A: ArrayLike, b: ArrayLike) -> NDArray:
        """Solve A x = b via our LU tool (chapter 8.5 exercises)."""
        try:
            from .tools.lu import lu_factor, lu_solve as _lu_solve  # type: ignore
        except Exception as e:  # pragma: no cover
            raise APIError(f"LU tools not available yet: {e}") from e

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        L, U = lu_factor(A)
        return _lu_solve(L, U, b)

    @_timeit
    def lu_inverse(self, A: ArrayLike) -> NDArray:
        """Compute A^{-1} via LU (and triangular inverses) for pedagogy."""
        try:
            from .tools.lu import lu_factor, lu_inverse  # type: ignore
        except Exception as e:  # pragma: no cover
            raise APIError(f"LU tools not available yet: {e}") from e

        A = np.asarray(A, dtype=float)
        L, U = lu_factor(A)
        return lu_inverse(L, U)

    # ---------------------------- Documentation Aids -----------------------------

    @_timeit
    def export_class_diagram(self, outdir: Union[str, Path] | None = None) -> Path:
        """
        Generate a class diagram of the `velocity` package.

        Implementation
        --------------
        Prefers `pyreverse` (ships with pylint) to emit PlantUML + DOT.
        Falls back to `pylint.pyreverse.main` invocation.

        Returns
        -------
        Path to the main diagram file (e.g., classes.dot or classes.uml).
        """
        outdir = Path(outdir or self.default_out)
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            # Local import to avoid a hard dependency in runtime unless used.
            from pylint.pyreverse.main import Run  # type: ignore
        except Exception as e:  # pragma: no cover
            raise APIError(
                "pyreverse (pylint) is required for diagram export. "
                "Install with `pip install pylint`."
            ) from e

        # pyreverse writes files into CWD; we call it on our package folder.
        pkg_dir = Path(__file__).resolve().parent
        # Run expects CLI-like args; we request DOT and PlantUML.
        Run([str(pkg_dir), "-o", "dot", "-p", "velocity"], exit=False)
        Run([str(pkg_dir), "-o", "plantuml", "-p", "velocity"], exit=False)

        # Move outputs to outdir for consistency.
        moved: Optional[Path] = None
        for name in ("classes.dot", "packages.dot", "classes.uml", "packages.uml"):
            src = Path.cwd() / name
            if src.exists():
                dest = outdir / name
                try:
                    dest.write_bytes(src.read_bytes())
                    src.unlink(missing_ok=True)
                    moved = dest if name.startswith("classes") else moved
                except Exception as e:  # pragma: no cover
                    self.log.warning("Could not move %s: %s", name, e)

        if not moved:
            raise APIError("pyreverse did not produce expected outputs.")
        return moved


# --------------------------------------------------------------------------- #
# Small typing helper so we can accept array-likes without importing numpy.typing
# (keeps early setup simple).
# --------------------------------------------------------------------------- #

ArrayLike = Union[Sequence[float], NDArray, List[float]]
