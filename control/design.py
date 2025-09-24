from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .core import RobotDynamics
from .utils import SecondOrderTuning

@dataclass(slots=True)
class PDController:
    kp: float
    kd: float
    def u(self, e: np.ndarray, ed: np.ndarray) -> np.ndarray:
        return -self.kd*ed - self.kp*e

@dataclass(slots=True)
class PIDController:
    kp: float; ki: float; kd: float
    _ei: np.ndarray | None = None
    def reset(self): self._ei = None
    def u(self, e: np.ndarray, ed: np.ndarray, dt: float) -> np.ndarray:
        if self._ei is None: self._ei = np.zeros_like(e)
        self._ei += e*dt
        return -self.kd*ed - self.kp*e - self.ki*self._ei

@dataclass(slots=True)
class ComputedTorquePD:
    """Implements (14.33) /(14.41) Q = D(q)(qdd_d - Kd e_dot - Kp e) + Cqd + g"""
    robot: RobotDynamics
    kp: np.ndarray
    kd: np.ndarray
    def torque(self, q, qd, qd_d, qdd_d, q_d) -> np.ndarray:
        e  = np.asarray(q)  - np.asarray(q_d)
        ed = np.asarray(qd) - np.asarray(qd_d)
        M  = self.robot.inertia(q)
        b  = self.robot.bias(q, qd)
        v  = qdd_d - self.kd*ed - self.kp*e
        return M @ v + b

def diagonal_pd_from_second_order(n: int, wn: float, zeta: float):
    s = SecondOrderTuning(wn, zeta)
    return s.kp()*np.ones(n), s.kd()*np.ones(n)
