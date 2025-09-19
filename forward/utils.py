# forward/utils.py
"""
Utility decorators and math helpers for forward kinematics.

Exports
-------
- Decorators:
    * timed(fn)        → records fn runtime (seconds) on wrapper.last_runtime
    * ensure_shape(*shape) → validates ndarray shape of the return value

- Numerical helpers:
    * TOL                    (default tolerance)
    * almost_equal(a, b)     → elementwise array closeness
    * clamp(x, lo, hi)
    * normalize(v)

- Lie/SE(3) helpers:
    * skew(v)         → 3×3 skew-symmetric from a 3-vector
    * vex(S)          → inverse of skew (vee operator)
    * hat_xi(ω, v)    → 4×4 se(3) matrix from screw (ω, v)
    * adjoint(T)      → 6×6 adjoint matrix of SE(3) transform
    * homogeneous(R, t) → 4×4 from (R, t)

- Validators:
    * is_rotation(R)
    * is_transform(T)

All helpers are NumPy-only to keep CI light and tests fast.
"""
from __future__ import annotations

import functools
import math
import time
from typing import Callable, Iterable, Optional, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Configuration / tolerances
# ---------------------------------------------------------------------------

#: Default absolute tolerance for equality checks.
TOL: float = 1e-9


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def timed(fn: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that measures wall-clock runtime of `fn`.

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
        wrapper.last_runtime = time.perf_counter() - t0
        return out
    wrapper.last_runtime = 0.0  # type: ignore[attr-defined]
    return wrapper


def ensure_shape(*shape: int) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """
    Decorator that validates the ndarray shape of the returned value.

    Parameters
    ----------
    shape : tuple[int]
        Expected shape, e.g., (4, 4).

    Raises
    ------
    ValueError
        If the returned object has a ``shape`` attribute that does not match.

    Examples
    --------
    >>> @ensure_shape(4, 4)
    ... def eye(): return np.eye(4)
    >>> eye().shape
    (4, 4)
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


# ---------------------------------------------------------------------------
# Simple numeric helpers
# ---------------------------------------------------------------------------

def almost_equal(a: np.ndarray, b: np.ndarray, *, atol: float = TOL, rtol: float = 1e-9) -> bool:
    """Elementwise closeness for arrays."""
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=rtol)


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a scalar to [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Return a unit-length copy of vector v; if ||v|| < eps, returns v unchanged.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


# ---------------------------------------------------------------------------
# Lie algebra & SE(3) helpers
# ---------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix [v]^ of a 3-vector.

    Parameters
    ----------
    v : (3,) array-like

    Returns
    -------
    (3,3) ndarray
    """
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=float)


def vex(S: np.ndarray) -> np.ndarray:
    """
    Vee operator (inverse of skew). Extracts the vector from a 3×3 skew-symmetric matrix.
    """
    S = np.asarray(S, dtype=float)
    if S.shape != (3, 3):
        raise ValueError("vex expects a 3x3 matrix")
    return np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2.0


def hat_xi(omega: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Build a 4×4 se(3) matrix from a screw (ω, v).

    Returns
    -------
    (4,4) ndarray
        [[ [ω]^, v ],
         [  0 , 0 ]]
    """
    Xi = np.zeros((4, 4), dtype=float)
    Xi[:3, :3] = skew(omega)
    Xi[:3, 3] = np.asarray(v, dtype=float).reshape(3)
    return Xi


def adjoint(T: np.ndarray) -> np.ndarray:
    """
    Adjoint representation Ad_T of an SE(3) transform T.

    Parameters
    ----------
    T : (4,4) ndarray

    Returns
    -------
    (6,6) ndarray
        [[R, 0],
         [p^ R, R]]
    where R is the rotation and p is the translation.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"adjoint expects 4x4, got {T.shape}")
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6), dtype=float)
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = skew(p) @ R
    return Ad


def homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compose a 4x4 homogeneous transform from (R, t).
    """
    H = np.eye(4, dtype=float)
    H[:3, :3] = np.asarray(R, float).reshape(3, 3)
    H[:3, 3] = np.asarray(t, float).reshape(3)
    return H


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def is_rotation(R: np.ndarray, *, atol: float = 1e-8) -> bool:
    """
    Heuristic check for a valid rotation matrix (orthonormal & det ≈ 1).
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    should_be_I = R.T @ R
    det = np.linalg.det(R)
    return np.allclose(should_be_I, np.eye(3), atol=atol) and math.isclose(det, 1.0, rel_tol=0.0, abs_tol=atol)


def is_transform(T: np.ndarray, *, atol: float = 1e-8) -> bool:
    """
    Heuristic check for a valid homogeneous transform.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        return False
    if not is_rotation(T[:3, :3], atol=atol):
        return False
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=atol):
        return False
    return True
