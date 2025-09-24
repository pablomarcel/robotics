from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Dict, Any
from .core import MassSpringDamper, Pendulum, Planar2R, SimulationResult
from .design import PDController, PIDController, ComputedTorquePD, diagonal_pd_from_second_order
from .utils import linearize
from .tools.trace import track

@dataclass(slots=True)
class SimAPI:
    @track("msd.sim")
    def simulate_msd(self, plant: MassSpringDamper, u_fun, t: tuple[float,float], x0: np.ndarray, dt: float=1e-3) -> SimulationResult:
        A, B, C, D = plant.A(), plant.B(), plant.C(), plant.D()
        def f(_t, x):
            u = np.array([u_fun(_t)])
            return (A @ x + (B @ u).ravel())
        ts = np.arange(t[0], t[1]+dt, dt)
        sol = solve_ivp(f, t, x0, t_eval=ts, rtol=1e-8, atol=1e-10)
        X = sol.y.T
        U = np.array([u_fun(tt) for tt in ts])[:,None]
        Y = (X @ C.T)
        return SimulationResult(ts, X, Y, U, {"A":A,"B":B,"C":C,"D":D})

    @track("pendulum.sim")
    def simulate_pendulum(self, plant: Pendulum, controller, t: tuple[float,float], x0: np.ndarray, ref: np.ndarray, dt: float=1e-3):
        def f(_t, x):
            q, qd = x[0], x[1]
            e  = np.array([q - ref[0]])
            ed = np.array([qd - ref[1]])
            if isinstance(controller, PIDController):
                u = controller.u(e, ed, dt)
            else:
                u = controller.u(e, ed)
            return plant.f(x, u)
        ts = np.arange(t[0], t[1]+dt, dt)
        sol = solve_ivp(f, t, x0, t_eval=ts, rtol=1e-8, atol=1e-10)
        X = sol.y.T; U = np.zeros((len(ts),1)); Y = X.copy()
        return SimulationResult(ts, X, Y, U, {"ref":ref})

    @track("pendulum.linearize")
    def linearize_pendulum(self, plant: Pendulum, x_op: np.ndarray, u_op: np.ndarray):
        A, B = linearize(lambda x,u: plant.f(x,u), x_op, u_op)
        C = np.eye(2); D = np.zeros((2,1))
        return {"A":A, "B":B, "C":C, "D":D}

    @track("robot.ctpd")
    def robot_computed_torque(self, robot: Planar2R|Any, q, qd, q_d, qd_d, qdd_d, wn=4.0, zeta=1.0):
        kp, kd = diagonal_pd_from_second_order(len(q), wn, zeta)
        ctl = ComputedTorquePD(robot, kp, kd)
        tau = ctl.torque(q, qd, qd_d, qdd_d, q_d)
        return {"tau": tau, "kp": kp, "kd": kd}
