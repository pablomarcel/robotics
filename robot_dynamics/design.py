from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from .core import Link, Joint, RobotModel


@dataclass(slots=True)
class DHParam:
    """Denavit–Hartenberg parameters for a single joint.

    Using standard DH: (a, alpha, d, theta); for a revolute joint `theta` is
    variable; for a prismatic joint `d` is variable.
    """
    a: float
    alpha: float
    d: float
    theta: float
    joint_type: str = "R"  # 'R' or 'P'

    def transform(self, q_i: float) -> np.ndarray:
        a, alpha, d, theta = self.a, self.alpha, self.d, self.theta
        if self.joint_type == "R":
            theta = q_i
        else:
            d = q_i
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ], dtype=float)


class DHChainBuilder:
    """Factory for common textbook models (e.g., 2R planar, 3R wrist)."""

    @staticmethod
    def planar_2r(l1: float, l2: float, m1: float, m2: float,
                  c1: float | None = None, c2: float | None = None) -> tuple[RobotModel, list[DHParam]]:
        c1 = c1 if c1 is not None else l1 / 2
        c2 = c2 if c2 is not None else l2 / 2
        links = [
            Link("L1", m1, np.array([c1, 0, 0], float), np.diag([0, 0, m1 * l1**2 / 12]), l1),
            Link("L2", m2, np.array([c2, 0, 0], float), np.diag([0, 0, m2 * l2**2 / 12]), l2),
        ]
        joints = [Joint("J1", "R", np.array([0, 0, 1.0])), Joint("J2", "R", np.array([0, 0, 1.0]))]
        model = RobotModel("Planar2R", links, joints)
        dh = [
            DHParam(a=l1, alpha=0.0, d=0.0, theta=0.0, joint_type="R"),
            DHParam(a=l2, alpha=0.0, d=0.0, theta=0.0, joint_type="R"),
        ]
        return model, dh