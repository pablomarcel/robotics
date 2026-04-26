# path/app.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable
import numpy as np

from .core import BoundaryConditions, SampledTrajectory
from .time import LSPB, QuinticTime
from .poly import CubicPoly, QuinticPoly, SepticPoly, LeastSquaresPoly
from .segment import Piecewise1D
from .spatial import ParabolicBlend3D, Harmonic1D, Cycloid1D, ComposeYofX
from .robot import Planar2R
from .rotation import AngleAxisPath
from .io import IOManager
# Optional, but nice to have available for diagrams from code:
# from .design import DesignManager


@dataclass
class PathPlannerApp:
    """
    High-level façade used by CLI and tests.

    Keeps the OO core focused and provides small factories/wrappers so
    tests and scripts don’t need to import from many modules. Nothing here
    does I/O implicitly; use `self.io` explicitly in CLI or tests.

    Notes
    -----
    - Uses `default_factory` to avoid mutable-default dataclass errors.
    - All returned objects are the core OO classes (CubicPoly, LSPB, etc.).
    - Helpers `sample_1d` / `sample_nd` return SampledTrajectory for plots/tests.
    - `poly_path` is the tiny convenience wrapper you asked for to assert
      textbook coefficients across 12.x examples.
    """
    io: IOManager = field(default_factory=IOManager)
    # design: DesignManager = field(default_factory=DesignManager)  # enable if you want programmatic diagrams here

    # ---- 1D polynomials -----------------------------------------------------

    def cubic(self, bc: BoundaryConditions) -> CubicPoly:
        """Cubic with endpoint position/velocity_kinematics (12.1–12.16)."""
        return CubicPoly(bc)

    def quintic(self, bc: BoundaryConditions) -> QuinticPoly:
        """Quintic with endpoint pos/vel/acc (12.50–12.57)."""
        return QuinticPoly(bc)

    def septic(self, bc: BoundaryConditions) -> SepticPoly:
        """Septic with zero jerk at ends (12.58–12.61)."""
        return SepticPoly(bc)

    def lspb(self, bc: BoundaryConditions, vmax: float | None = None, amax: float | None = None) -> LSPB:
        """Linear segment with parabolic blends (trapezoid/triangle) (12.26–12.49, 12.62–12.74)."""
        return LSPB(bc, vmax=vmax, amax=amax)

    # ---- Least-squares / fitting -------------------------------------------

    def ls_poly(
        self,
        t: Iterable[float],
        q: Iterable[float],
        degree: int,
        bc: BoundaryConditions | None = None,
    ) -> LeastSquaresPoly:
        """
        Least-squares polynomial fit (12.113–12.121).

        Parameters
        ----------
        t, q : sample vectors
        degree : polynomial degree
        bc : optional BoundaryConditions (used mainly to carry t0/tf for consistency)

        Returns
        -------
        LeastSquaresPoly
        """
        t_arr = np.asarray(t, dtype=float)
        q_arr = np.asarray(q, dtype=float)
        bc_eff = bc or BoundaryConditions(t_arr[0], t_arr[-1], q_arr[0], q_arr[-1])
        ls = LeastSquaresPoly(bc_eff, degree=degree, t_samples=t_arr, q_samples=q_arr)
        ls.fit()
        return ls

    # ---- Spatial / compositions --------------------------------------------

    def parabolic_blend3d(self, r0, r1, r2, t0, t1, t2, tblend) -> ParabolicBlend3D:
        """3D parabolic corner blend (12.147–12.165, 12.171–12.176)."""
        return ParabolicBlend3D(r0, r1, r2, t0, t1, t2, tblend)

    def harmonic(self, t0: float, tf: float, q0: float, qf: float, w: float) -> Harmonic1D:
        """Harmonic rest-to-rest (12.132–12.142)."""
        return Harmonic1D.fit_rest2rest(t0, tf, q0, qf, w)

    def cycloid(self, t0: float, tf: float, q0: float, qf: float) -> Cycloid1D:
        """Cycloid rest-to-rest (12.143–12.146)."""
        return Cycloid1D(t0, tf, q0, qf)

    def compose_y_of_x(self, fx: Callable[[np.ndarray], np.ndarray], Xt: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        """
        Compose Y = f(X(t)) used in 12.166–12.170 and circle bypass (12.237–12.239).

        Returns a callable Y(t).
        """
        return ComposeYofX(fx).compose(Xt)

    # ---- Robot (2R examples, 12.181–12.230, 12.231+) ------------------------

    def planar2r(self, l1: float, l2: float, elbow: str = "up") -> Planar2R:
        """Instantiate the simple planar 2R model used in many examples."""
        return Planar2R(l1, l2, elbow)

    # ---- Rotation (SO(3), 12.251–12.283) -----------------------------------

    def angle_axis_path(self, R0: np.ndarray, Rf: np.ndarray) -> AngleAxisPath:
        """Axis-angle path between two rotation_kinematics matrices (Rodrigues / linear in angle)."""
        return AngleAxisPath(np.asarray(R0, float), np.asarray(Rf, float))

    # ---- Sampling helpers ---------------------------------------------------

    def sample_1d(self, traj, t: Iterable[float]) -> SampledTrajectory:
        """
        Sample a 1D trajectory into (t, q, qd, qdd).

        Returns
        -------
        SampledTrajectory
        """
        t_arr = np.asarray(t, dtype=float)
        return SampledTrajectory(t_arr, traj.q(t_arr), traj.qd(t_arr), traj.qdd(t_arr))

    def sample_nd(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, t: Iterable[float]) -> SampledTrajectory:
        """
        Wrap already-evaluated ND arrays into a SampledTrajectory for consistent IO.
        Mostly useful for ParabolicBlend3D which already returns SampledTrajectory,
        but handy if you generate arrays externally.
        """
        t_arr = np.asarray(t, dtype=float)
        return SampledTrajectory(t_arr, np.asarray(q), np.asarray(qd), np.asarray(qdd))

    # ---- Tiny convenience for tests: poly_path ------------------------------

    def poly_path(
        self,
        q0: float,
        qf: float,
        t0: float,
        tf: float,
        order: int,
        *,
        qd0: float = 0.0,
        qdf: float = 0.0,
        qdd0: float = 0.0,
        qddf: float = 0.0,
        samples: int = 200,
    ) -> dict:
        """
        Convenience wrapper that builds {3:Cubic, 5:Quintic, 7:Septic}, then returns:
        {
          "a": coefficients (if polynomial),
          "t": t grid,
          "q": q(t),
          "qd": qd(t),
          "qdd": qdd(t)
        }
        Useful for unit tests asserting textbook coefficients (12.x).
        """
        bc = BoundaryConditions(t0, tf, q0, qf, qd0, qdf, qdd0, qddf)
        cls_map = {3: CubicPoly, 5: QuinticPoly, 7: SepticPoly}
        if order not in cls_map:
            raise ValueError(f"Unsupported order {order}; expected one of {sorted(cls_map)}")
        traj = cls_map[order](bc)
        t = np.linspace(t0, tf, samples)
        q = traj.q(t); qd = traj.qd(t); qdd = traj.qdd(t)
        coeffs = getattr(traj, "coefficients", lambda: None)()
        return {"a": None if coeffs is None else np.asarray(coeffs),
                "t": t, "q": q, "qd": qd, "qdd": qdd}
