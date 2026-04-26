# applied_dynamics/utils.py
"""
Utility decorators, math helpers, and path constants for **applied_dynamics dynamics**.

Highlights
----------
- Decorators:
    * timed(fn)                   → records fn runtime on wrapper.last_runtime
    * ensure_shape(*shape)        → validate ndarray shape
    * ensure_sympy_shape(r, c)    → validate sympy Matrix shape
    * timeit(fn)                  → back-compat: returns (result, dt)

- Numeric helpers:
    * TOL, almost_equal(a, b), clamp(x, lo, hi), normalize(v)

- Lie/SE(3) helpers commonly used across dynamics sims:
    * skew(v), vex(S), homogeneous(R, t)

- Paths & I/O helpers:
    * APPLIED_ROOT, IN_DIR, OUT_DIR
    * ensure_outfile(path)
    * Result dataclass (light result carrier)
"""
from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar

import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APPLIED_ROOT = Path(__file__).resolve().parent
IN_DIR = APPLIED_ROOT / "in"
OUT_DIR = APPLIED_ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IN_DIR.mkdir(parents=True, exist_ok=True)

def ensure_outfile(path: Path) -> Path:
    """Ensure parent directory exists and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

# ---------------------------------------------------------------------------
# Configuration / tolerances
# ---------------------------------------------------------------------------

#: Default absolute tolerance for numerical comparisons.
TOL: float = 1e-9

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def timed(fn: Callable[..., T]) -> Callable[..., T]:
    """
    Measure wall-clock runtime of `fn`.

    The wrapper exposes a float attribute ``last_runtime`` with the duration.

    Examples
    --------
    >>> @timed
    ... def f(x): return x**2
    >>> _ = f(3)
    >>> f.last_runtime >= 0.0
    True
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        wrapper.last_runtime = time.perf_counter() - t0  # type: ignore[attr-defined]
        return out
    wrapper.last_runtime = 0.0  # type: ignore[attr-defined]
    return wrapper


def timeit(fn: Callable[..., T]) -> Callable[..., Tuple[T, float]]:
    """
    Back-compat decorator matching the old API: returns (result, dt).

    Prefer :func:`timed` for new code; keep this to avoid breaking callers.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Tuple[T, float]:
        t0 = time.perf_counter()
        res = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        return res, dt
    return wrapper


def ensure_shape(*shape: int) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """
    Validate the ndarray shape of the return value.

    Parameters
    ----------
    shape : tuple[int]
        Expected shape, e.g., (6,) or (4, 4).
    """
    def deco(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> np.ndarray:
            out = fn(*args, **kwargs)
            if not hasattr(out, "shape") or tuple(out.shape) != tuple(shape):
                raise ValueError(f"{fn.__name__} expected shape {shape}, got {getattr(out, 'shape', None)}")
            return out
        return wrapper
    return deco


def ensure_sympy_shape(rows: int, cols: int) -> Callable[[Callable[..., sp.Matrix]], Callable[..., sp.Matrix]]:
    """
    Validate the SymPy Matrix shape of the return value.

    Examples
    --------
    >>> @ensure_sympy_shape(2, 1)
    ... def f(): return sp.Matrix([1, 2])
    """
    def deco(fn: Callable[..., sp.Matrix]) -> Callable[..., sp.Matrix]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> sp.Matrix:
            out = fn(*args, **kwargs)
            if not isinstance(out, sp.MatrixBase) or tuple(out.shape) != (rows, cols):
                raise ValueError(f"{fn.__name__} expected sympy Matrix shape {(rows, cols)}, got {getattr(out, 'shape', None)}")
            return sp.Matrix(out)
        return wrapper
    return deco

# ---------------------------------------------------------------------------
# Simple numeric helpers
# ---------------------------------------------------------------------------

def almost_equal(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray,
                 *, atol: float = TOL, rtol: float = 1e-9) -> bool:
    """Elementwise closeness for arrays."""
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=rtol)


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a scalar to [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def normalize(v: Sequence[float] | np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a unit-length copy of vector v; if ||v|| < eps, returns v unchanged."""
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n

# ---------------------------------------------------------------------------
# Lie / SE(3) helpers (handy for many dynamics/kinematics utilities)
# ---------------------------------------------------------------------------

def skew(v: Sequence[float] | np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]^ of a 3-vector."""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z,  y],
                     [z,  0.0, -x],
                     [-y,  x,  0.0]], dtype=float)


def vex(S: np.ndarray) -> np.ndarray:
    """Vee operator (inverse_kinematics of skew). Extract vector from a 3×3 skew-symmetric matrix."""
    S = np.asarray(S, dtype=float)
    if S.shape != (3, 3):
        raise ValueError("vex expects a 3x3 matrix")
    return np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2.0


def homogeneous(R: np.ndarray, t: Sequence[float] | np.ndarray) -> np.ndarray:
    """Compose a 4×4 homogeneous transform from (R, t)."""
    H = np.eye(4, dtype=float)
    H[:3, :3] = np.asarray(R, float).reshape(3, 3)
    H[:3, 3] = np.asarray(t, float).reshape(3)
    return H

# ---------------------------------------------------------------------------
# Lightweight result carrier
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """Lightweight result carrier for computations."""
    name: str
    data: Dict[str, Any]
    notes: str = ""

# ---------------------------------------------------------------------------

__all__ = [
    # paths
    "APPLIED_ROOT", "IN_DIR", "OUT_DIR", "ensure_outfile",
    # decorators
    "timed", "timeit", "ensure_shape", "ensure_sympy_shape",
    # helpers
    "TOL", "almost_equal", "clamp", "normalize", "skew", "vex", "homogeneous",
    # result
    "Result",
]
