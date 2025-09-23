from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, Iterable, Tuple, Dict, Any
import numpy as np

class PathError(Exception):
    """Domain-specific exception for path-planning operations."""

@dataclass(slots=True)
class BoundaryConditions:
    """Boundary conditions for 1D trajectories at t0, tf."""
    t0: float
    tf: float
    q0: float
    qf: float
    qd0: float = 0.0
    qdf: float = 0.0
    qdd0: float = 0.0
    qddf: float = 0.0
    jerk0: float | None = None
    jerkf: float | None = None

class Differentiable(ABC):
    """Interface for objects that provide value and time-derivatives."""
    @abstractmethod
    def q(self, t: np.ndarray | float) -> np.ndarray | float: ...
    @abstractmethod
    def qd(self, t: np.ndarray | float) -> np.ndarray | float: ...
    @abstractmethod
    def qdd(self, t: np.ndarray | float) -> np.ndarray | float: ...

@dataclass
class Trajectory1D(Differentiable, ABC):
    """Base for scalar trajectories."""
    bc: BoundaryConditions

    @abstractmethod
    def coefficients(self) -> np.ndarray:
        """Polynomial coefficients lowest→highest power or other param set."""

@dataclass
class TrajectoryND(Differentiable, ABC):
    """Base for vector trajectories (ℝ^n)."""
    dims: int

@dataclass
class SampledTrajectory:
    """Holds sampled q,qd,qdd over time; convenient for IO/plots."""
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray

    def as_dict(self) -> Dict[str, Any]:
        return {"t": self.t, "q": self.q, "qd": self.qd, "qdd": self.qdd}

def ensure_1d_array(x: Iterable[float] | float) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))

def finite_diff(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple central finite differences for testing/plots."""
    dt = np.gradient(t)
    xd = np.gradient(x, t)
    xdd = np.gradient(xd, t)
    return xd, xdd
