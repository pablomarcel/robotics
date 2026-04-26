from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Callable, Tuple
from .core import Trajectory1D, SampledTrajectory

@dataclass
class Piecewise1D(Trajectory1D):
    """Join multiple Trajectory1D segments (12.82–12.112, 12.151-like)."""
    segments: List[Trajectory1D] = None  # each has its own bc within [t_i, t_{i+1}]

    def q(self, t):
        return self._apply('q', t)
    def qd(self, t):
        return self._apply('qd', t)
    def qdd(self, t):
        return self._apply('qdd', t)

    def _apply(self, fn: str, t):
        t = np.asarray(t, float)
        out = np.empty_like(t)
        for seg in self.segments:
            m = (t >= seg.bc.t0) & (t <= seg.bc.tf + 1e-12)
            if m.any():
                out[m] = getattr(seg, fn)(t[m])
        return out

    def coefficients(self):
        return np.array([getattr(s, "coefficients", lambda: None)() for s in self.segments], dtype=object)
