from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .core import BoundaryConditions, SampledTrajectory
from .time import LSPB, QuinticTime
from .poly import CubicPoly, QuinticPoly, SepticPoly, LeastSquaresPoly
from .segment import Piecewise1D
from .spatial import ParabolicBlend3D, Harmonic1D, Cycloid1D, ComposeYofX
from .robot import Planar2R
from .rotation import AngleAxisPath
from .io import IOManager

@dataclass
class PathPlannerApp:
    """High-level façade used by CLI and tests."""
    io: IOManager = IOManager()

    # ---- 1D polynomials
    def cubic(self, bc: BoundaryConditions) -> CubicPoly:
        return CubicPoly(bc)
    def quintic(self, bc: BoundaryConditions) -> QuinticPoly:
        return QuinticPoly(bc)
    def septic(self, bc: BoundaryConditions) -> SepticPoly:
        return SepticPoly(bc)
    def lspb(self, bc: BoundaryConditions, vmax=None, amax=None) -> LSPB:
        return LSPB(bc, vmax=vmax, amax=amax)

    # ---- Least-squares fit
    def ls_poly(self, t, q, degree: int, bc: BoundaryConditions|None=None) -> LeastSquaresPoly:
        ls = LeastSquaresPoly(bc or BoundaryConditions(t[0], t[-1], q[0], q[-1]), degree=degree,
                              t_samples=np.asarray(t), q_samples=np.asarray(q))
        ls.fit(); return ls

    # ---- Spatial: parabolic blend, harmonic, cycloid
    def parabolic_blend3d(self, r0,r1,r2,t0,t1,t2,tblend):
        return ParabolicBlend3D(r0,r1,r2,t0,t1,t2,tblend)
    def harmonic(self, t0, tf, q0, qf, w): return Harmonic1D.fit_rest2rest(t0, tf, q0, qf, w)
    def cycloid(self, t0, tf, q0, qf): return Cycloid1D(t0, tf, q0, qf)
    def compose_y_of_x(self, fx, Xt): return ComposeYofX(fx).compose(Xt)

    # ---- Robot: analytic 2R
    def planar2r(self, l1, l2, elbow="up"): return Planar2R(l1, l2, elbow)

    # ---- Rotation
    def angle_axis_path(self, R0, Rf): return AngleAxisPath(R0, Rf)

    # ---- Helpers
    def sample_1d(self, traj, t):
        t=np.asarray(t); return SampledTrajectory(t, traj.q(t), traj.qd(t), traj.qdd(t))
