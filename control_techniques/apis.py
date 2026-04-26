# control_techniques/apis.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from .core import MassSpringDamper, Pendulum, Planar2R, SimulationResult
from .design import PIDController, ComputedTorquePD, diagonal_pd_from_second_order
from .utils import linearize
from .tools.trace import track


def _time_grid(t_span: Tuple[float, float], dt: float) -> np.ndarray:
    """Return a t_eval grid within [t0, tf], never exceeding tf.

    - Guarantees t_eval[0] == t0.
    - Ensures the last sample equals tf (appending it exactly if needed).
    - Raises ValueError on invalid (dt <= 0) or reversed span.
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if tf < t0:
        raise ValueError("t_span must be nondecreasing")
    # number of steps so that t_k = t0 + k*dt <= tf
    n = int(np.floor((tf - t0) / dt)) + 1
    ts = t0 + np.arange(n, dtype=float) * dt
    if ts[-1] < tf:
        ts = np.append(ts, tf)
    return ts


@dataclass(slots=True)
class SimAPI:
    """Simulation/analysis façade for the Control module (14.1–14.118)."""

    @track("msd.sim")
    def simulate_msd(
        self,
        plant: MassSpringDamper,
        u_fun,
        t: Tuple[float, float],
        x0: np.ndarray,
        dt: float = 1e-3,
    ) -> SimulationResult:
        """Simulate mass–spring–damper with input u_fun(t)."""
        A, B, C, D = plant.A(), plant.B(), plant.C(), plant.D()

        def f(_t, x):
            u = np.array([u_fun(_t)], dtype=float)
            return (A @ x + (B @ u).ravel())

        ts = _time_grid(t, dt)
        sol = solve_ivp(f, t, x0, t_eval=ts, rtol=1e-8, atol=1e-10)
        X = sol.y.T
        U = np.array([u_fun(tt) for tt in ts], dtype=float)[:, None]
        Y = (X @ C.T)
        return SimulationResult(ts, X, Y, U, {"A": A, "B": B, "C": C, "D": D})

    @track("pendulum.sim")
    def simulate_pendulum(
        self,
        plant: Pendulum,
        controller,
        t: Tuple[float, float],
        x0: np.ndarray,
        ref: np.ndarray,
        dt: float = 1e-3,
    ) -> SimulationResult:
        """Simulate pendulum with an external PD/PID-like controller object.

        Controller must implement:

        - PIDController.u(e, ed, dt) or

        - PD-like .u(e, ed)

        """
        def f(_t, x):
            q, qd = float(x[0]), float(x[1])
            e = np.array([q - float(ref[0])], dtype=float)
            ed = np.array([qd - float(ref[1])], dtype=float)
            if isinstance(controller, PIDController):
                u = controller.u(e, ed, dt)
            else:  # PD-like interface
                u = controller.u(e, ed)
            return plant.f(x, u)

        ts = _time_grid(t, dt)
        sol = solve_ivp(f, t, x0, t_eval=ts, rtol=1e-8, atol=1e-10)
        X = sol.y.T
        U = np.zeros((len(ts), 1), dtype=float)  # placeholder; control_techniques is internal
        Y = X.copy()
        return SimulationResult(ts, X, Y, U, {"ref": ref})

    @track("pendulum.linearize")
    def linearize_pendulum(self, plant: Pendulum, x_op: np.ndarray, u_op: np.ndarray):
        """Numerical linearization around operating point (14.82–14.86)."""
        A, B = linearize(lambda x, u: plant.f(x, u), x_op, u_op)
        C = np.eye(2); D = np.zeros((2, 1))
        return {"A": A, "B": B, "C": C, "D": D}

    @track("robot_dynamics.ctpd")
    def robot_computed_torque(
        self,
        robot: Planar2R | Any,
        q,
        qd,
        q_d,
        qd_d,
        qdd_d,
        wn: float = 4.0,
        zeta: float = 1.0,
    ):
        """Computed-torque PD: Q = D(q)(qdd_d - Kd e_dot - Kp e) + Cqd + g."""
        kp, kd = diagonal_pd_from_second_order(len(q), wn, zeta)
        ctl = ComputedTorquePD(robot, kp, kd)
        tau = ctl.torque(q, qd, qd_d, qdd_d, q_d)
        return {"tau": tau, "kp": kp, "kd": kd}
