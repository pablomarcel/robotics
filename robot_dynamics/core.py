from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Tuple, Protocol
import numpy as np


@dataclass(slots=True)
class Link:
    """Rigid link with mass and inertia in its body frame.

    Attributes
    ----------
    name: str
        Identifier.
    mass: float
        Mass `m_i`.
    com: np.ndarray
        3-vector of center of mass in the link frame `B_i`.
    inertia: np.ndarray
        3x3 inertia matrix about `C_i` expressed in `B_i`.
    length: float | None
        Convenience for many planar examples.
    """
    name: str
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    length: Optional[float] = None


@dataclass(slots=True)
class Joint:
    """One-DoF joint (R or P).

    Parameters
    ----------
    name: str
    type: str
        'R' or 'P'.
    axis: np.ndarray
        3-vector along joint axis expressed in parent frame.
    """
    name: str
    type: str
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


@dataclass(slots=True)
class State:
    """Joint-space state container."""
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray | None = None


class RobotModel:
    """Serial-chain robot_dynamics model using DH or arbitrary transforms.

    This class stores topology and inertial parameters and offers utilities to
    interface with different dynamics engines.
    """
    def __init__(self, name: str, links: List[Link], joints: List[Joint]):
        assert len(links) == len(joints), "links and joints must match"
        self.name = name
        self.links = links
        self.joints = joints

    # --- basic queries ---------------------------------------------------
    @property
    def dof(self) -> int:
        return len(self.joints)

    def masses(self) -> np.ndarray:
        return np.array([ln.mass for ln in self.links], dtype=float)

    def inertias(self) -> List[np.ndarray]:
        return [ln.inertia for ln in self.links]


# --- small math helpers used by several modules ---------------------------

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def hat_2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)