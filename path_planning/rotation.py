from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    u = np.asarray(axis, float)
    n = np.linalg.norm(u)
    if n == 0:
        return np.eye(3)
    u = u / n
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], float)
    I = np.eye(3)
    return I + np.sin(theta) * ux + (1 - np.cos(theta)) * (ux @ ux)

def _axis_angle_from_R(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Stable axis-angle from rotation_kinematics matrix; robust at 0 and π."""
    R = np.asarray(R, float)
    # clamp trace to [-1,3]
    tr = np.clip(np.trace(R), -1.0, 3.0)
    angle = np.arccos((tr - 1) / 2.0)
    if np.isclose(angle, 0.0, atol=1e-12):
        return np.array([1.0, 0.0, 0.0]), 0.0
    if np.isclose(angle, np.pi, atol=1e-9):
        # Robust axis for 180-deg: use diagonal entries
        x = np.sqrt(max(0.0, (R[0, 0] + 1) / 2))
        y = np.sqrt(max(0.0, (R[1, 1] + 1) / 2))
        z = np.sqrt(max(0.0, (R[2, 2] + 1) / 2))
        axis = np.array([x, y, z])
        # Fix signs using off-diagonals
        if R[0, 1] < 0: axis[1] = -axis[1]
        if R[0, 2] < 0: axis[2] = -axis[2]
        if np.linalg.norm(axis) == 0:
            axis = np.array([1.0, 0.0, 0.0])
        return axis / np.linalg.norm(axis), angle
    # generic case
    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1],
    ])
    axis = axis / (2*np.sin(angle))
    return axis / np.linalg.norm(axis), angle

@dataclass
class AngleAxisPath:
    """SO(3) path_planning via axis-angle (12.251–12.273)."""
    R0: np.ndarray
    Rf: np.ndarray
    axis: np.ndarray | None = None
    angle: float | None = None

    def __post_init__(self):
        Rrel = self.R0.T @ self.Rf
        self.axis, self.angle = _axis_angle_from_R(Rrel)

    def R(self, s: np.ndarray) -> np.ndarray:
        """
        Interpolate linearly in angle: s ∈ [0,1].
        Returns array shape (N,3,3).
        """
        s = np.atleast_1d(np.asarray(s, float))
        mats = [self.R0 @ rodrigues(self.axis, float(si)*self.angle) for si in s]
        return np.stack(mats, axis=0)
