from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .core import BoundaryConditions, Trajectory1D
from .utils import solve

@dataclass
class LSPB(Trajectory1D):
    """
    Linear segment with parabolic blends (aka trapezoidal/triangular velocity_kinematics).
    Covers 12.26–12.49 and 12.62–12.74 behavior.
    """
    vmax: float | None = None
    amax: float | None = None

    def __post_init__(self):
        if self.vmax is None and self.amax is None:
            raise ValueError("Provide either vmax or amax for LSPB")

    def _params(self):
        t0, tf, q0, qf = self.bc.t0, self.bc.tf, self.bc.q0, self.bc.qf
        D = qf - q0
        T = tf - t0
        if self.amax is None:
            V = float(self.vmax)
            ta = min(T/2.0, D/V/2.0) if V>0 else T/2.0
        else:
            a = float(self.amax)
            ta = min(T/2.0, np.sqrt(D/a))
            V = a * ta * 2.0 / 2.0 * (D/abs(D)) if D!=0 else 0.0
        tb = T - 2*ta
        return ta, tb, V, D, T

    def q(self, t):
        q0 = self.bc.q0
        t = np.asarray(t, dtype=float)
        t_ = t - self.bc.t0
        ta, tb, V, D, T = self._params()
        out = np.empty_like(t_)
        # accel
        m = t_ <= ta
        out[m] = q0 + 0.5*(V/ta)*(t_[m]**2)
        # cruise
        m2 = (t_ > ta) & (t_ <= ta+tb)
        out[m2] = q0 + 0.5*V*ta + V*(t_[m2]-ta)
        # decel
        m3 = t_ > ta+tb
        td = t_[m3] - (ta+tb)
        out[m3] = q0 + 0.5*V*ta + V*tb + V*td - 0.5*(V/ta)*(td**2)
        return out

    def qd(self, t):
        t = np.asarray(t, dtype=float)
        t_ = t - self.bc.t0
        ta, tb, V, D, T = self._params()
        out = np.empty_like(t_)
        out[t_ <= ta] = (V/ta)*t_[t_<=ta]
        m = (t_ > ta) & (t_ <= ta+tb)
        out[m] = V
        m2 = t_ > ta+tb
        out[m2] = V - (V/ta)*(t_[m2]-(ta+tb))
        return out

    def qdd(self, t):
        t = np.asarray(t, dtype=float)
        t_ = t - self.bc.t0
        ta, tb, V, D, T = self._params()
        a = V/ta if ta>0 else 0.0
        out = np.zeros_like(t_)
        out[t_ <= ta] = a
        out[t_ > ta+tb] = -a
        return out

    def coefficients(self):  # not a polynomial; return key params
        return np.array(self._params())

@dataclass
class QuinticTime(Trajectory1D):
    """Quintic time-scaling with zero end vel/acc (12.50–12.57, 12.220–12.227)."""
    def coefficients(self):
        t0, tf, q0, qf = self.bc.t0, self.bc.tf, self.bc.q0, self.bc.qf
        T = tf - t0
        a0 = q0
        a1 = 0.0
        a2 = 0.0
        a3 = 10*(qf-q0)/T**3
        a4 = -15*(qf-q0)/T**4
        a5 = 6*(qf-q0)/T**5
        return np.array([a0,a1,a2,a3,a4,a5])

    def q(self, t):
        a = self.coefficients(); t = np.asarray(t)-self.bc.t0
        return a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
    def qd(self, t):
        a = self.coefficients(); t = np.asarray(t)-self.bc.t0
        return a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
    def qdd(self, t):
        a = self.coefficients(); t = np.asarray(t)-self.bc.t0
        return 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3
