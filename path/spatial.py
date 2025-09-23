from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable
from .core import TrajectoryND, SampledTrajectory
from .time import LSPB, QuinticTime

@dataclass
class ParabolicBlend3D(TrajectoryND):
    """3D corner blend with constant-accel parabolas (12.147–12.165, 12.171–12.176)."""
    r0: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    t0: float
    t1: float
    t2: float
    tblend: float  # t'
    def __init__(self, r0,r1,r2,t0,t1,t2,tblend):
        super().__init__(3); self.r0,self.r1,self.r2 = map(lambda x: np.asarray(x,float),(r0,r1,r2))
        self.t0,self.t1,self.t2,self.tblend = t0,t1,t2,tblend

    def sample(self, t: np.ndarray) -> SampledTrajectory:
        t = np.asarray(t,float)
        q = np.zeros((t.size,3)); qd=np.zeros_like(q); qdd=np.zeros_like(q)
        # line 1: r0->r1, line 2: r1->r2
        v1 = (self.r1-self.r0)/(self.t1-self.t0)
        v2 = (self.r2-self.r1)/(self.t2-self.t1)
        # blend polynomial per 12.161–12.165
        for i,ti in enumerate(t):
            if ti <= self.t1 - self.tblend:
                tau = (ti-self.t0); q[i]=self.r0+v1*tau; qd[i]=v1; qdd[i]=0
            elif ti >= self.t1 + self.tblend:
                tau = (ti-self.t1); # begin at r1
                r1p = self.r1 + v1*(self.tblend)
                # maintain continuity by offsetting
                q[i]=r1p + v2*(tau - self.tblend); qd[i]=v2; qdd[i]=0
            else:
                ta = self.tblend
                tau = ti-(self.t1-ta)
                # constant acceleration from v1 to v2 in 2*ta seconds
                ac = (v2 - v1) / (2*ta)
                q[i] = self.r1 - v1*ta + v1*tau + 0.5*ac*(tau**2)      # 12.162 style
                qd[i]= v1 + ac*tau
                qdd[i]= ac
        return SampledTrajectory(t,q,qd,qdd)

@dataclass
class Harmonic1D:
    """q(t)=a0+a1 cos(ωt)+a3 sin(ωt) with rest-to-rest (12.132–12.142)."""
    a0: float; a1: float; a3: float; w: float
    @classmethod
    def fit_rest2rest(cls, t0, tf, q0, qf, w):
        A = np.array([
            [1, np.cos(w*t0), np.sin(w*t0)],
            [1, np.cos(w*tf), np.sin(w*tf)],
            [0, -w*np.sin(w*t0), w*np.cos(w*t0)]
        ], float)
        b = np.array([q0,qf,0.0], float)
        a0,a1,a3 = np.linalg.solve(A,b)
        return cls(a0,a1,a3,w)
    def q(self,t): t=np.asarray(t); return self.a0 + self.a1*np.cos(self.w*t) + self.a3*np.sin(self.w*t)
    def qd(self,t): t=np.asarray(t); return -self.a1*self.w*np.sin(self.w*t) + self.a3*self.w*np.cos(self.w*t)
    def qdd(self,t): t=np.asarray(t); return -self.a1*self.w**2*np.cos(self.w*t) - self.a3*self.w**2*np.sin(self.w*t)

@dataclass
class Cycloid1D:
    """Cycloid rest-to-rest (12.143–12.146)."""
    t0: float; tf: float; q0: float; qf: float
    def q(self,t):
        t=np.asarray(t); T=self.tf-self.t0
        return 0.5*(self.qf+self.q0) - 0.5*(self.qf-self.q0)*np.cos(np.pi*(t-self.t0)/T)
    def qd(self,t):
        t=np.asarray(t); T=self.tf-self.t0
        return 0.5*(self.qf-self.q0)*(np.pi/T)*np.sin(np.pi*(t-self.t0)/T)
    def qdd(self,t):
        t=np.asarray(t); T=self.tf-self.t0
        return 0.5*(self.qf-self.q0)*((np.pi/T)**2)*np.cos(np.pi*(t-self.t0)/T)

@dataclass
class ComposeYofX:
    """Compose Y=f(X(t)) for lines/circles/etc. (12.166–12.170, 12.237–12.239)."""
    fx: callable
    def compose(self, Xt: callable) -> callable:
        return lambda t: self.fx(Xt(t))
