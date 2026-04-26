from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Planar2R:
    """Simple 2R planar arm used across examples (12.181–12.230)."""
    l1: float
    l2: float
    elbow: str = "up"  # or "down"

    # FK
    def fk(self, th1: np.ndarray, th2: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        x = self.l1*np.cos(th1) + self.l2*np.cos(th1+th2)
        y = self.l1*np.sin(th1) + self.l2*np.sin(th1+th2)
        return x, y

    # IK
    def ik(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        c2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2*self.l1*self.l2)
        c2 = np.clip(c2, -1.0, 1.0)
        s2 = np.sqrt(1 - c2**2) * (1 if self.elbow=="up" else -1)
        th2 = np.arctan2(s2, c2)
        k1 = self.l1 + self.l2*c2
        k2 = self.l2*s2
        th1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        return th1, th2
