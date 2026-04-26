# path/spatial.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from .core import TrajectoryND, SampledTrajectory

# ---------------------------------------------------------------------------
# 3D parabolic blend between two straight segments (12.147–12.165, 12.171–12.176)
# ---------------------------------------------------------------------------

@dataclass
class ParabolicBlend3D(TrajectoryND):
    """3D corner blend with constant-accel parabolas between two straight segments.

    Segments:
        r0 -> r1 over [t0, t1]
        r1 -> r2 over [t1, t2]

    The blend of duration `tblend` (= t′ in the text) is centered at t1, i.e.
    [t1 - tblend, t1 + tblend]. Acceleration is constant (vector) within the
    blend, yielding C¹ continuity of velocity_kinematics across the corner.
    """
    r0: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    t0: float
    t1: float
    t2: float
    tblend: float  # t′

    def __init__(self, r0, r1, r2, t0, t1, t2, tblend):
        super().__init__(3)
        self.r0 = np.asarray(r0, float)
        self.r1 = np.asarray(r1, float)
        self.r2 = np.asarray(r2, float)
        self.t0 = float(t0); self.t1 = float(t1); self.t2 = float(t2)
        self.tblend = float(tblend)

    def _v1_v2(self):
        v1 = (self.r1 - self.r0) / (self.t1 - self.t0)
        v2 = (self.r2 - self.r1) / (self.t2 - self.t1)
        return v1, v2

    def _piece(self, t: float):
        v1, v2 = self._v1_v2()
        ta = self.tblend
        if t <= self.t1 - ta:
            tau = t - self.t0
            q = self.r0 + v1 * tau
            qd = v1
            qdd = np.zeros(3)
        elif t >= self.t1 + ta:
            # position at the end of blend along first segment:
            r_blend_end = self.r1 + v1 * ta
            tau = t - (self.t1 + ta)
            q = r_blend_end + v2 * tau
            qd = v2
            qdd = np.zeros(3)
        else:
            # inside the blend window [t1 - ta, t1 + ta]
            tau = t - (self.t1 - ta)
            ac = (v2 - v1) / (2 * ta)  # constant vector acceleration over 2*ta
            # position continuity: start from r1 - v1*ta (which equals r at t1-ta)
            q = (self.r1 - v1 * ta) + v1 * tau + 0.5 * ac * (tau ** 2)
            qd = v1 + ac * tau
            qdd = ac
        return q, qd, qdd

    # -- TrajectoryND interface
    def q(self, t):
        t = np.asarray(t, float)
        if t.ndim == 0:
            return self._piece(float(t))[0]
        return np.stack([self._piece(float(tt))[0] for tt in t], axis=0)

    def qd(self, t):
        t = np.asarray(t, float)
        if t.ndim == 0:
            return self._piece(float(t))[1]
        return np.stack([self._piece(float(tt))[1] for tt in t], axis=0)

    def qdd(self, t):
        t = np.asarray(t, float)
        if t.ndim == 0:
            return self._piece(float(t))[2]
        return np.stack([self._piece(float(tt))[2] for tt in t], axis=0)

    def sample(self, t: np.ndarray) -> SampledTrajectory:
        t = np.asarray(t, float)
        return SampledTrajectory(t, self.q(t), self.qd(t), self.qdd(t))

# ---------------------------------------------------------------------------
# Harmonic rest-to-rest (12.132–12.142)
# ---------------------------------------------------------------------------

@dataclass
class Harmonic1D:
    """q(t) = a0 + a1 cos(ωt) + a3 sin(ωt) with rest-to-rest boundary conditions."""
    a0: float
    a1: float
    a3: float
    w: float

    @classmethod
    def fit_rest2rest(cls, t0, tf, q0, qf, w):
        A = np.array([
            [1, np.cos(w*t0), np.sin(w*t0)],
            [1, np.cos(w*tf), np.sin(w*tf)],
            [0, -w*np.sin(w*t0), w*np.cos(w*t0)],
        ], float)
        b = np.array([q0, qf, 0.0], float)
        a0, a1, a3 = np.linalg.solve(A, b)
        return cls(a0, a1, a3, w)

    def q(self, t):
        t = np.asarray(t); return self.a0 + self.a1*np.cos(self.w*t) + self.a3*np.sin(self.w*t)

    def qd(self, t):
        t = np.asarray(t); return -self.a1*self.w*np.sin(self.w*t) + self.a3*self.w*np.cos(self.w*t)

    def qdd(self, t):
        t = np.asarray(t); return -self.a1*self.w**2*np.cos(self.w*t) - self.a3*self.w**2*np.sin(self.w*t)

# ---------------------------------------------------------------------------
# Cycloid rest-to-rest (12.143–12.146)
# ---------------------------------------------------------------------------

@dataclass
class Cycloid1D:
    """Closed-form cycloidal motion_kinematics between (t0,q0) and (tf,qf)."""
    t0: float
    tf: float
    q0: float
    qf: float

    def q(self, t):
        t = np.asarray(t); T = self.tf - self.t0
        return 0.5*(self.qf + self.q0) - 0.5*(self.qf - self.q0)*np.cos(np.pi*(t - self.t0)/T)

    def qd(self, t):
        t = np.asarray(t); T = self.tf - self.t0
        return 0.5*(self.qf - self.q0)*(np.pi/T)*np.sin(np.pi*(t - self.t0)/T)

    def qdd(self, t):
        t = np.asarray(t); T = self.tf - self.t0
        return 0.5*(self.qf - self.q0)*((np.pi/T)**2)*np.cos(np.pi*(t - self.t0)/T)

# ---------------------------------------------------------------------------
# Compose Y = f(X(t)) (12.166–12.170, 12.237–12.239)
# ---------------------------------------------------------------------------

@dataclass
class ComposeYofX:
    """Compose Y(t) = f(X(t)) for analytic shapes like lines/circles and a time-law X(t)."""
    fx: Callable[[np.ndarray], np.ndarray]

    def compose(self, Xt: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        return lambda t: self.fx(Xt(t))
