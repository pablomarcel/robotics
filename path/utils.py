from __future__ import annotations
import importlib
import numpy as np

def have_scipy() -> bool:
    try:
        importlib.import_module("scipy")
        return True
    except Exception:
        return False

def vandermonde(t0: float, tf: float, degree: int, bc_types: tuple[str,...]) -> np.ndarray:
    """
    Build a (n+1)×(n+1) system matrix for polynomial boundary conditions.
    bc_types is a tuple like ('q(t0)','qd(t0)','q(tf)','qd(tf)', ...).
    """
    def row(t: float, k: int, n: int):
        # kth derivative row at time t (k=0..n)
        coeff = np.zeros(n+1)
        for p in range(k, n+1):
            # d^k/dt^k (t^p) = p*(p-1)*..*(p-k+1) t^{p-k}
            c = 1.0
            for s in range(k): c *= (p - s)
            coeff[p] = (t ** (p - k)) * c
        return coeff

    n = degree
    rows = []
    for bc in bc_types:
        if bc.startswith('q(') and bc.endswith('t0)'):
            rows.append(row(t0, 0, n))
        elif bc.startswith('qd(') and bc.endswith('t0)'):
            rows.append(row(t0, 1, n))
        elif bc.startswith('qdd(') and bc.endswith('t0)'):
            rows.append(row(t0, 2, n))
        elif bc.startswith('q(') and bc.endswith('tf)'):
            rows.append(row(tf, 0, n))
        elif bc.startswith('qd(') and bc.endswith('tf)'):
            rows.append(row(tf, 1, n))
        elif bc.startswith('qdd(') and bc.endswith('tf)'):
            rows.append(row(tf, 2, n))
        else:
            raise ValueError(f"Unsupported bc token: {bc}")
    return np.vstack(rows)

def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, b)
