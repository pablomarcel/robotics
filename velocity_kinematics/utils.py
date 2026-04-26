# velocity_kinematics/utils.py
"""
Utility math & decorators for the Velocity Kinematics Toolkit.

Contents
--------
- Decorators: @timed
- Linear algebra helpers: skew(), pinv_damped()
- SO(3)/SE(3) utilities: rotx/roty/rotz, trotx/troty/trotz, transl, mmul
- Small helpers: r2t(), t2r(), normalize()

Design goals
------------
- Zero non-stdlib deps beyond NumPy.
- Deterministic, tiny, and easy to test.
- Explicit shapes and docstrings to support future Sphinx docs.
"""

from __future__ import annotations

import functools
import time
from typing import Iterable, Sequence, Tuple

import numpy as np

Array = np.ndarray


# ------------------------------- Decorators ---------------------------------- #

def timed(fn):
    """
    Time a function call and attach the elapsed seconds to `fn.last_runtime_s`.

    Example
    -------
    >>> @timed
    ... def f(x):
    ...     return x**2
    >>> f(3)
    9
    >>> assert f.last_runtime_s is not None
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        wrapper.last_runtime_s = time.perf_counter() - t0
        return out
    wrapper.last_runtime_s = None  # type: ignore[attr-defined]
    return wrapper


# ------------------------------ Linear algebra ------------------------------- #

def skew(v: Array) -> Array:
    """
    Return the 3×3 skew-symmetric matrix [v]_× such that [v]_× w = v × w.

    Parameters
    ----------
    v : (3,) array

    Returns
    -------
    (3,3) array
    """
    v = np.asarray(v, dtype=float).reshape(3)
    x, y, z = v
    return np.array([[0.0, -z,   y],
                     [  z, 0.0, -x],
                     [ -y,  x, 0.0]], dtype=float)


def pinv_damped(A: Array, lam: float = 0.0) -> Array:
    """
    Damped pseudoinverse (Tikhonov / Levenberg–Marquardt) via SVD.

    For A = U diag(s) V^T, returns:
        A^#_λ = V diag( s_i / (s_i^2 + λ^2) ) U^T

    Notes
    -----
    - When λ → 0 this approaches the Moore–Penrose pseudoinverse.
    - Stable near singularities when λ > 0.
    - Works for both tall and wide A.

    Parameters
    ----------
    A : (m, n) array_like
        Matrix to pseudo-invert.
    lam : float, default 0.0
        Damping parameter λ ≥ 0.

    Returns
    -------
    (n, m) ndarray
        Damped pseudoinverse of A.
    """
    A = np.asarray(A, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    lam2 = float(lam) ** 2
    # s / (s^2 + lam^2) applied along singular values
    s_damped = s / (s**2 + lam2) if lam2 > 0.0 else np.divide(
        1.0, s, out=np.zeros_like(s), where=(s != 0.0)
    )
    return (Vt.T * s_damped) @ U.T


def normalize(v: Array, eps: float = 1e-12) -> Array:
    """Return v / ||v||, guarding against division by zero."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v if n < eps else (v / n)


# ------------------------------- SO(3) / SE(3) ------------------------------- #

def rotx(alpha: float) -> Array:
    """SO(3) rotation_kinematics about x-axis."""
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]], dtype=float)


def roty(beta: float) -> Array:
    """SO(3) rotation_kinematics about y-axis."""
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[  c, 0.0,   s],
                     [0.0, 1.0, 0.0],
                     [ -s, 0.0,   c]], dtype=float)


def rotz(theta: float) -> Array:
    """SO(3) rotation_kinematics about z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[  c,  -s, 0.0],
                     [  s,   c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def r2t(R: Array, p: Sequence[float] | None = None) -> Array:
    """
    Lift an SO(3) matrix (and optional position) to SE(3).

    Parameters
    ----------
    R : (3,3)
    p : optional (3,)

    Returns
    -------
    (4,4) array
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def t2r(T: Array) -> Tuple[Array, Array]:
    """
    Split an SE(3) matrix into (R, p).

    Returns
    -------
    R : (3,3)
    p : (3,)
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    return T[:3, :3], T[:3, 3].copy()


def trotx(alpha: float) -> Array:
    """SE(3) rotation_kinematics about x-axis."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = rotx(alpha)
    return T


def troty(beta: float) -> Array:
    """SE(3) rotation_kinematics about y-axis."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = roty(beta)
    return T


def trotz(theta: float) -> Array:
    """SE(3) rotation_kinematics about z-axis."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = rotz(theta)
    return T


def transl(x: float, y: float, z: float) -> Array:
    """SE(3) pure translation."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.array([x, y, z], dtype=float)
    return T


def mmul(*mats: Array) -> Array:
    """
    Left-to-right matrix multiply convenience.

    Example
    -------
    >>> M = mmul(transl(1,0,0), trotz(0.3), trotx(-0.1))
    """
    if not mats:
        raise ValueError("mmul requires at least one matrix")
    out = np.asarray(mats[0], dtype=float)
    for M in mats[1:]:
        out = out @ np.asarray(M, dtype=float)
    return out
