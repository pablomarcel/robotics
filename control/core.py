from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any
import numpy as np

class NonlinearSystem(Protocol):
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: ...
    def h(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: ...

@dataclass(slots=True)
class SimulationResult:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    meta: Dict[str, Any]

# ---------- Canonical plants matching the chapter ----------
@dataclass(slots=True)
class MassSpringDamper:
    m: float; c: float; k: float
    def A(self): return np.array([[0, 1],[-self.k/self.m, -self.c/self.m]], float)
    def B(self): return np.array([[0],[1/self.m]], float)
    def C(self): return np.array([[1,0]], float)
    def D(self): return np.array([[0]], float)

@dataclass(slots=True)
class Pendulum:
    m: float; l: float; I: float; c: float; g: float = 9.81
    # State: x=[theta, theta_dot], input u=[Q]
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        th, thd = x; Q = float(u[0])
        thdd = (Q - self.c*thd - self.m*self.g*self.l*np.sin(th))/self.I
        return np.array([th, thd + 0]) * 0 + np.array([thd, thdd])  # (14.81)
    def h(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x  # measure [θ, θdot]

# Minimal robot abstraction for computed-torque
class RobotDynamics(Protocol):
    def inertia(self, q: np.ndarray) -> np.ndarray: ...
    def bias(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Return C(q,qd)@qd + g(q)."""

@dataclass(slots=True)
class Planar2R:
    """Simple 2R using Craig DH; closed-form D,C,g (lightweight)."""
    m1: float = 1.0; m2: float = 1.0
    l1: float = 1.0; l2: float = 1.0
    r1: float = 0.5; r2: float = 0.5
    I1: float = 0.1; I2: float = 0.1
    g: float = 9.81

    def inertia(self, q: np.ndarray) -> np.ndarray:
        q1, q2 = q
        a = self.I1 + self.I2 + self.m1*self.r1**2 + self.m2*(self.l1**2 + self.r2**2)
        b = self.m2*self.l1*self.r2*np.cos(q2)
        c = self.I2 + self.m2*self.r2**2
        return np.array([[a + 2*b, c + b],[c + b, c]], float)

    def coriolis_mat(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        q1, q2 = q; q1d, q2d = qd
        h = -self.m2*self.l1*self.r2*np.sin(q2)
        return np.array([[h*q2d, h*(q1d+q2d)],[-h*q1d, 0.0]], float)

    def grav(self, q: np.ndarray) -> np.ndarray:
        q1, q2 = q
        g1 = (self.m1*self.r1 + self.m2*self.l1)*self.g*np.cos(q1) + self.m2*self.r2*self.g*np.cos(q1+q2)
        g2 = self.m2*self.r2*self.g*np.cos(q1+q2)
        return np.array([g1, g2], float)

    def bias(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        return self.coriolis_mat(q, qd) @ qd + self.grav(q)
