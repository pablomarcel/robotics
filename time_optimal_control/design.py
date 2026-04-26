# time/design.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# -------- 2R kinematics helpers (straight line or arc in XY) --------

@dataclass(slots=True)
class Planar2RGeom:
    l1: float = 1.0
    l2: float = 1.0

    def ik(self, x: float, y: float, elbow_up: bool = True) -> np.ndarray:
        """Return [theta, phi] for a reachable (x,y) with standard 2R IK."""
        c2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2*self.l1*self.l2)
        c2 = np.clip(c2, -1.0, 1.0)
        s2 = np.sqrt(max(0.0, 1 - c2**2))
        if not elbow_up: s2 = -s2
        phi = np.arctan2(s2, c2)
        k1 = self.l1 + self.l2 * c2
        k2 = self.l2 * s2
        theta = np.arctan2(y, x) - np.arctan2(k2, k1)
        return np.array([theta, phi])

    def path_line_y_const(self, y: float, x0: float, x1: float, n: int = 200, elbow_up=True) -> np.ndarray:
        xs = np.linspace(x0, x1, n)
        qs = [self.ik(x, y, elbow_up=elbow_up) for x in xs]
        return np.vstack(qs)  # shape (n,2)

    def path_arc(self, cx: float, cy: float, r: float, ang0: float, ang1: float, n: int = 200, elbow_up=True) -> np.ndarray:
        th = np.linspace(ang0, ang1, n)
        xs, ys = cx + r*np.cos(th), cy + r*np.sin(th)
        qs = [self.ik(x, y, elbow_up=elbow_up) for x, y in zip(xs, ys)]
        return np.vstack(qs)
