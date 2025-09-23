from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    u = np.asarray(axis, float); u = u/np.linalg.norm(u)
    ux = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]], float)
    I = np.eye(3)
    return I + np.sin(theta)*ux + (1-np.cos(theta))*(ux@ux)

@dataclass
class AngleAxisPath:
    """SO(3) path via axis-angle (12.251–12.273)."""
    R0: np.ndarray
    Rf: np.ndarray
    axis: np.ndarray | None = None
    angle: float | None = None

    def __post_init__(self):
        Rrel = self.R0.T @ self.Rf
        angle = np.arccos(np.clip(0.5*(np.trace(Rrel)-1), -1, 1))
        if self.axis is None:
            if np.isclose(angle, 0): self.axis = np.array([1,0,0]); self.angle=0; return
            self.axis = (1/(2*np.sin(angle))) * np.array([
                Rrel[2,1]-Rrel[1,2],
                Rrel[0,2]-Rrel[2,0],
                Rrel[1,0]-Rrel[0,1],
            ])
        self.angle = float(angle)

    def R(self, t: np.ndarray) -> np.ndarray:
        """Linear schedule in angle (12.260); piecewise LSPB could be used from time.py."""
        t = np.atleast_1d(np.asarray(t,float))
        mats = []
        for s in t:
            Rrel = rodrigues(self.axis, s*self.angle)
            mats.append(self.R0 @ Rrel)
        return np.stack(mats, axis=0)
