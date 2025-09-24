from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Callable

@dataclass(slots=True)
class SecondOrderTuning:
    """Convenience map for (14.39–14.40)."""
    wn: float
    zeta: float
    def kp(self) -> float: return self.wn**2
    def kd(self) -> float: return 2*self.zeta*self.wn

def linearize(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
              x0: np.ndarray, u0: np.ndarray, eps: float = 1e-6
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Numerical Jacobians A=∂f/∂x, B=∂f/∂u at (x0,u0)."""
    x0 = np.asarray(x0, float); u0 = np.asarray(u0, float)
    n, m = x0.size, u0.size
    f0 = np.asarray(f(x0, u0), float)
    A = np.zeros((n, n)); B = np.zeros((n, m))
    for i in range(n):
        dx = np.zeros_like(x0); dx[i] = eps
        A[:, i] = (f(x0+dx, u0) - f0)/eps
    for j in range(m):
        du = np.zeros_like(u0); du[j] = eps
        B[:, j] = (f(x0, u0+du) - f0)/eps
    return A, B
