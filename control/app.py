# control/app.py
"""
Orchestration façade used by CLI/tests. Keeps objects small & testable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .core import MassSpringDamper, Pendulum, Planar2R
from .design import PDController, PIDController
from .apis import SimAPI


@dataclass(slots=True)
class ControlApp:
    # Use default_factory to avoid a mutable default at class definition time.
    api: SimAPI = field(default_factory=SimAPI)

    # Quick factories
    def msd(self, m: float = 1.0, c: float = 0.8, k: float = 10.0) -> MassSpringDamper:
        return MassSpringDamper(m, c, k)

    def pendulum(self, m: float = 1.0, l: float = 0.35, I: float = 0.07, c: float = 0.01) -> Pendulum:
        return Pendulum(m, l, I, c)

    def planar2r(self) -> Planar2R:
        return Planar2R()

    # Ready scenarios
    def run_pendulum_pid_at_pi_over_2(self, T: float = 3.0):
        """Simulate PID about θd = π/2 (maps to 14.80–14.88)."""
        pl = self.pendulum()
        x0 = np.array([np.pi / 2, 0.0])
        ref = np.array([np.pi / 2, 0.0])
        ctl = PIDController(kp=30, ki=5, kd=10)
        return self.api.simulate_pendulum(pl, ctl, (0.0, T), x0, ref, dt=1e-3)
