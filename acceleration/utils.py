# acceleration/utils.py
"""
Utility decorators and math helpers for **acceleration kinematics**.

What you get
------------
- Decorators:
    * timed(fn)              → records fn runtime on wrapper.last_runtime
    * ensure_shape(*shape)   → validates ndarray shape of the return value

- Numerical helpers:
    * TOL, almost_equal(a, b), clamp(x, lo, hi), normalize(v)
    * asvec(x, n)            → strict vector coercion (nice for tests)

- Lie / SO(3) / SE(3):
    * skew(v), vex(S)
    * homogeneous(R, t)      → 4x4 from (R, t)
    * adjoint(T)             → 6x6 Ad_T  (useful if tests check frame switches)

- Acceleration building blocks:
    * S_from(alpha, omega)   → α̃ + ω̃²
    * omega_from_Rdot(R, Rdot)                (ω from Ṙ Rᵀ)
    * alpha_from_Rddot(R, Rdot, Rddot)        (α from R̈ Rᵀ)
    * classic_accel(alpha, omega, r)          (α×r + ω×(ω×r))

- Finite-difference helper (for backends/tests):
    * jdot_qdot_fd(J_fn, q, qd, eps=1e-8)
      where J_fn(q) returns a Jacobian; computes (∂J/∂q · q̇) q̇ via directional FD

All functions are NumPy-only to keep CI light and tests fast.
"""

from __future__ import annotations

import functools
import time
from typing import Callable, Iterable, Sequence, TypeVar

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
        Expected shape, e.g., (3,) or (3, 3).
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


def asvec(x: Iterable[float] | np.ndarray, n: int) -> np.ndarray:
    """
    Strict 1-D vector coercion with length check (great for unit tests).
    """
    v = np.asarray(x, float).reshape(-1)
    if v.size != n:
        raise ValueError(f"Expected vector of length {n}, got {v.size}")
    return v


# ---------------------------------------------------------------------------
# Lie algebra & SE(3) helpers
# ---------------------------------------------------------------------------

def skew(v: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix [v]^ of a 3-vector.

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
    return np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]], float) * 0.5


def homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compose a 4×4 homogeneous transform from (R, t).
    """
    H = np.eye(4, dtype=float)
    H[:3, :3] = np.asarray(R, float).reshape(3, 3)
    H[:3, 3] = np.asarray(t, float).reshape(3)
    return H


def adjoint(T: np.ndarray) -> np.ndarray:
    """
    Adjoint representation Ad_T of an SE(3) transform T.

    Returns
    -------
    (6,6) ndarray
        [[R, 0],
         [p^ R, R]]
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


# ---------------------------------------------------------------------------
# Acceleration building blocks
# ---------------------------------------------------------------------------

def S_from(alpha: Sequence[float] | np.ndarray, omega: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Return the acceleration transform matrix S = α̃ + ω̃².
    """
    A = skew(asvec(alpha, 3))
    W = skew(asvec(omega, 3))
    return A + W @ W


def omega_from_Rdot(R: np.ndarray, Rdot: np.ndarray) -> np.ndarray:
    """
    ω from the identity Ṙ Rᵀ = ω̃ (body angular velocity).
    """
    R = np.asarray(R, float).reshape(3, 3)
    Rdot = np.asarray(Rdot, float).reshape(3, 3)
    omega_hat = Rdot @ R.T
    # Enforce skew by removing symmetric part (robust to small numeric noise)
    omega_hat = 0.5 * (omega_hat - omega_hat.T)
    return vex(omega_hat)


def alpha_from_Rddot(R: np.ndarray, Rdot: np.ndarray, Rddot: np.ndarray) -> np.ndarray:
    """
    α from R̈ Rᵀ = α̃ + ω̃²  ⇒  α̃ = R̈ Rᵀ − ω̃².
    """
    R = np.asarray(R, float).reshape(3, 3)
    Rdot = np.asarray(Rdot, float).reshape(3, 3)
    Rddot = np.asarray(Rddot, float).reshape(3, 3)
    omega = omega_from_Rdot(R, Rdot)
    alpha_hat = Rddot @ R.T - skew(omega) @ skew(omega)
    alpha_hat = 0.5 * (alpha_hat - alpha_hat.T)
    return vex(alpha_hat)


def classic_accel(alpha: Sequence[float], omega: Sequence[float], r: Sequence[float]) -> np.ndarray:
    """
    Classic rigid-body point acceleration:
        a = α×r + ω×(ω×r)   (tangential + centripetal)
    """
    α = asvec(alpha, 3)
    ω = asvec(omega, 3)
    r = asvec(r, 3)
    return np.cross(α, r) + np.cross(ω, np.cross(ω, r))


# ---------------------------------------------------------------------------
# Finite-difference helper (useful for tests / tiny backends)
# ---------------------------------------------------------------------------

def jdot_qdot_fd(J_fn: Callable[[np.ndarray], np.ndarray],
                 q: Sequence[float], qd: Sequence[float], eps: float = 1e-8) -> np.ndarray:
    """
    Compute (J̇(q, q̇)) q̇ via directional finite differences:

        J̇(q) q̇ ≈ [J(q + ε q̇) - J(q - ε q̇)] / (2ε)  · q̇

    Parameters
    ----------
    J_fn : callable
        Function returning the Jacobian at q (shape m×n).
    q : array-like (n,)
    qd: array-like (n,)
    eps : float
        Finite-difference step.

    Returns
    -------
    (m,) ndarray
    """
    q = asvec(q, np.asarray(q, float).size)
    qd = asvec(qd, q.size)
    Jp = np.asarray(J_fn(q + eps * qd), float)
    Jm = np.asarray(J_fn(q - eps * qd), float)
    Jdot_dir = (Jp - Jm) / (2.0 * eps)   # m×n
    return Jdot_dir @ qd


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "TOL",
    # Decorators
    "timed",
    "ensure_shape",
    # Numerics
    "almost_equal",
    "clamp",
    "normalize",
    "asvec",
    # Lie / SE(3)
    "skew",
    "vex",
    "homogeneous",
    "adjoint",
    # Accel blocks
    "S_from",
    "omega_from_Rdot",
    "alpha_from_Rddot",
    "classic_accel",
    # FD helper
    "jdot_qdot_fd",
]
