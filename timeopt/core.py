# time/core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Optional, Tuple, List, Callable
import abc
import numpy as np


# ---------- Decorators (imported by others via utils) ----------

def requires(pkg: str):
    """Decorator to lazily assert an optional dependency is present."""
    def deco(fn):
        def inner(*a, **k):
            try:
                __import__(pkg)
            except Exception as e:
                raise RuntimeError(f"Optional dependency '{pkg}' is required: {e}")
            return fn(*a, **k)
        return inner
    return deco


# ---------- Data containers ----------

@dataclass(slots=True)
class SolveRequest:
    """Generic request to solve a time-optimal control problem."""
    name: str
    params: Dict[str, Any]
    out_dir: str = "time/out"


@dataclass(slots=True)
class SolveResult:
    """Standard result payload for all problems."""
    ok: bool
    message: str
    data: Dict[str, Any]


# ---------- Abstract base types ----------

class TimeOptimalProblem(abc.ABC):
    """
    Base class for all problems. Subclasses must implement:
      - build()  : construct internal model/transcription
      - solve()  : run the optimizer / algorithm
      - result() : package SolveResult (JSON-serializable)
    """
    def __init__(self, name: str):
        self.name = name
        self._built = False
        self._solved = False
        self._result: Optional[SolveResult] = None

    @abc.abstractmethod
    def build(self) -> "TimeOptimalProblem": ...

    @abc.abstractmethod
    def solve(self) -> "TimeOptimalProblem": ...

    @abc.abstractmethod
    def result(self) -> SolveResult: ...

    # convenience
    def run(self) -> SolveResult:
        return self.build().solve().result()


# ---------- Protocols for robotics problems ----------

class Dynamics2R(Protocol):
    """Protocol for planar 2R dynamics: D(q), C-like term, and G(q)."""
    def D(self, q: np.ndarray) -> np.ndarray: ...
    def Hdotdot(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray: ...
    def G(self, q: np.ndarray) -> np.ndarray: ...


# ---------- Utility math helpers ----------

def wrap_list(x) -> List[float]:
    return list(map(float, x)) if isinstance(x, (list, tuple, np.ndarray)) else [float(x)]
