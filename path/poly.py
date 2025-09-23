from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .core import BoundaryConditions, Trajectory1D
from .utils import vandermonde, solve

class PolyBase(Trajectory1D):
    degree: int = 0
    _coeff: np.ndarray | None = None

    def _build(self, bc_tokens: tuple[str,...], rhs: np.ndarray) -> np.ndarray:
        A = vandermonde(self.bc.t0, self.bc.tf, self.degree, bc_tokens)
        c = solve(A, rhs)
        self._coeff = c
        return c

    def coefficients(self) -> np.ndarray:
        assert self._coeff is not None, "Call .q() once to trigger build or call coefficients() after init"
        return self._coeff

    def _eval(self, t, deriv: int) -> np.ndarray:
        if self._coeff is None:
            raise RuntimeError("coefficients not computed")
        t = np.asarray(t, dtype=float)
        T = t
        out = np.zeros_like(T)
        n = self.degree
        c = self._coeff
        # Horner with derivative
        for p in range(n, -1, -1):
            if p < deriv: continue
            # term: c[p] * p*(p-1)*..*(p-deriv+1) * t^{p-deriv}
            coeff = c[p]
            for s in range(deriv): coeff *= (p - s)
            out = out * T + coeff
        # finally multiply by t^{0}? Horner did it.
        # We built evaluation for monic basis; works.
        # shift to (t - t0) domain to improve conditioning:
        return out

@dataclass
class CubicPoly(PolyBase):
    """Cubic with endpoint pos/vel (12.1–12.16)."""
    degree: int = 3
    def _ensure(self):
        bc = self.bc
        tokens = ('q(t0)','qd(t0)','q(tf)','qd(tf)')
        rhs = np.array([bc.q0, bc.qd0, bc.qf, bc.qdf], dtype=float)
        self._build(tokens, rhs)

    def q(self, t):
        if self._coeff is None: self._ensure()
        t = np.asarray(t, dtype=float)
        # evaluate in original t (not shifted) using numpy.polyval style (lowest→highest)
        c = self._coeff
        return c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3
    def qd(self, t):
        c = self._coeff if self._coeff is not None else self._ensure() or self._coeff
        return c[1] + 2*c[2]*t + 3*c[3]*t**2
    def qdd(self, t):
        c = self._coeff if self._coeff is not None else self._ensure() | self._coeff  # type: ignore
        c = self._coeff
        return 2*c[2] + 6*c[3]*np.asarray(t)

@dataclass
class QuinticPoly(PolyBase):
    """Quintic with pos/vel/acc at ends (12.50–12.57)."""
    degree: int = 5
    def _ensure(self):
        bc = self.bc
        tokens = ('q(t0)','qd(t0)','qdd(t0)','q(tf)','qd(tf)','qdd(tf)')
        rhs = np.array([bc.q0, bc.qd0, bc.qdd0, bc.qf, bc.qdf, bc.qddf], float)
        self._build(tokens, rhs)
    def q(self,t):
        if self._coeff is None: self._ensure()
        c = self._coeff; t=np.asarray(t)
        return sum(c[i]*t**i for i in range(6))
    def qd(self,t):
        c = self._coeff if self._coeff is not None else self._ensure() or self._coeff
        t=np.asarray(t); return c[1]+2*c[2]*t+3*c[3]*t**2+4*c[4]*t**3+5*c[5]*t**4
    def qdd(self,t):
        c = self._coeff if self._coeff is not None else self._ensure() or self._coeff
        t=np.asarray(t); return 2*c[2]+6*c[3]*t+12*c[4]*t**2+20*c[5]*t**3

@dataclass
class SepticPoly(PolyBase):
    """Septic with zero jerk at ends (12.58–12.61)."""
    degree: int = 7
    def _ensure(self):
        bc = self.bc
        # enforce q,qd,qdd,jerk at both ends
        tokens = ('q(t0)','qd(t0)','qdd(t0)','q(tf)','qd(tf)','qdd(tf)')
        rhs = np.array([bc.q0, bc.qd0, bc.qdd0, bc.qf, bc.qdf, bc.qddf], float)
        # add jerk zeros
        # expand A and rhs with t0 jerk row and tf jerk row
        from .utils import vandermonde
        A = vandermonde(bc.t0, bc.tf, self.degree, tokens)
        # append jerk rows:
        def jerk_row(t: float, n: int=7):
            r = np.zeros(n+1)
            for p in range(3, n+1):
                r[p] = p*(p-1)*(p-2)*(t**(p-3))
            return r
        A = np.vstack([A, jerk_row(bc.t0), jerk_row(bc.tf)])
        rhs = np.concatenate([rhs, [0.0, 0.0]])
        c = np.linalg.solve(A, rhs)
        self._coeff = c

    def q(self,t):
        if self._coeff is None: self._ensure()
        c=self._coeff; t=np.asarray(t); return sum(c[i]*t**i for i in range(8))
    def qd(self,t):
        c=self._coeff if self._coeff is not None else self._ensure() or self._coeff
        t=np.asarray(t); return c[1]+2*c[2]*t+3*c[3]*t**2+4*c[4]*t**3+5*c[5]*t**4+6*c[6]*t**5+7*c[7]*t**6
    def qdd(self,t):
        c=self._coeff if self._coeff is not None else self._ensure() or self._coeff
        t=np.asarray(t); return 2*c[2]+6*c[3]*t+12*c[4]*t**2+20*c[5]*t**3+30*c[6]*t**4+42*c[7]*t**5

@dataclass
class LeastSquaresPoly(Trajectory1D):
    """Least-squares polynomial approximating points (12.113–12.121)."""
    degree: int = 5
    t_samples: np.ndarray = None  # set by user
    q_samples: np.ndarray = None
    _coeff: np.ndarray | None = None

    def fit(self) -> np.ndarray:
        t = np.asarray(self.t_samples, float)
        q = np.asarray(self.q_samples, float)
        V = np.vander(t, N=self.degree+1, increasing=True)
        c, *_ = np.linalg.lstsq(V, q, rcond=None)
        self._coeff = c
        return c

    def coefficients(self) -> np.ndarray:
        return self._coeff if self._coeff is not None else self.fit()

    def q(self, t):
        c = self.coefficients(); t=np.asarray(t)
        return sum(c[i]*t**i for i in range(len(c)))
    def qd(self, t):
        c = self.coefficients(); t=np.asarray(t)
        return sum((i*c[i]) * t**(i-1) for i in range(1,len(c)))
    def qdd(self, t):
        c = self.coefficients(); t=np.asarray(t)
        return sum((i*(i-1)*c[i]) * t**(i-2) for i in range(2,len(c)))
