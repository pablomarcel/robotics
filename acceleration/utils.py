# acceleration/utils.py
"""
Utility decorators and math helpers for **acceleration kinematics**.
[...docstring unchanged...]
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

    Preserves complex dtype when `x` is complex. This is important for
    complex-step differentiation paths that temporarily promote to complex.
    """
    arr = np.asarray(x)
    dtype = float if np.isrealobj(arr) else complex
    v = np.asarray(arr, dtype=dtype).reshape(-1)
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
    (3,3) ndarray with dtype preserved from `v` (supports complex-step).
    """
    vv = np.asarray(v).reshape(3)
    x, y, z = vv
    dt = vv.dtype
    return np.array([[0,   -z,   y],
                     [z,    0,  -x],
                     [-y,   x,   0]], dtype=dt)


def vex(S: np.ndarray) -> np.ndarray:
    """
    Vee operator (inverse_kinematics of skew). Extracts the vector from a 3×3 skew-symmetric matrix.

    Preserves dtype (supports complex-step).
    """
    S = np.asarray(S)
    if S.shape != (3, 3):
        raise ValueError("vex expects a 3x3 matrix")
    return np.array([S[2, 1] - S[1, 2],
                     S[0, 2] - S[2, 0],
                     S[1, 0] - S[0, 1]], dtype=S.dtype) * 0.5


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
# Acceleration building blocks (BODY-FRAME ω, α)
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
    BODY-FRAME angular velocity from the kinematic identity:

        Rᵀ Ṙ = [ω]^   (ω expressed in the body frame)

    Compute in extended precision for robustness, then return float64.
    """
    R = np.asarray(R, np.longdouble).reshape(3, 3)
    Rdot = np.asarray(Rdot, np.longdouble).reshape(3, 3)
    omega_hat = R.T @ Rdot
    omega_hat = 0.5 * (omega_hat - omega_hat.T)  # keep skew part only
    return vex(omega_hat).astype(float)


def alpha_from_Rddot(R: np.ndarray, Rdot: np.ndarray, Rddot: np.ndarray) -> np.ndarray:
    """
    BODY-FRAME angular acceleration from the kinematic identities:

        Rᵀ Ṙ = ω^
        Rᵀ R̈ = α^ + ω^ ω^

    Numerically robust computation in extended precision:
        α^ = skew( Rᵀ R̈ - ω^ ω^ )
    """
    R = np.asarray(R, np.longdouble).reshape(3, 3)
    Rdot = np.asarray(Rdot, np.longdouble).reshape(3, 3)
    Rddot = np.asarray(Rddot, np.longdouble).reshape(3, 3)

    # ω^ = skew(Rᵀ Ṙ)
    W = R.T @ Rdot
    W = 0.5 * (W - W.T)

    # α^ = skew( Rᵀ R̈ - ω^ ω^ )
    A = R.T @ Rddot - (W @ W)
    alpha_hat = 0.5 * (A - A.T)
    return vex(alpha_hat)


def classic_accel(alpha: Sequence[float], omega: Sequence[float], r: Sequence[float]) -> np.ndarray:
    """
    Classic rigid-body point acceleration (body frame):
        a = α×r + ω×(ω×r)

    Compute using the *double cross* in extended precision to reduce cancellation
    relative to forming W @ (W @ r). Return float64.
    """
    α = np.asarray(alpha, np.longdouble).reshape(3)
    ω = np.asarray(omega, np.longdouble).reshape(3)
    rr = np.asarray(r,     np.longdouble).reshape(3)
    a = np.cross(α, rr) + np.cross(ω, np.cross(ω, rr))
    return np.asarray(a, float)


# ---------------------------------------------------------------------------
# Finite-difference helper (useful for tests / tiny backends)
# ---------------------------------------------------------------------------

def jdot_qdot_fd(J_fn: Callable[[np.ndarray], np.ndarray],
                 q: Sequence[float], qd: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    """
    Compute (J̇(q, q̇)) q̇ via a high-accuracy 5-point central stencil along q̇:

        d/dt J(q(t))|_{t=0}  with  q(t) = q + t q̇
        ≈ [-J(q+2h q̇) + 8 J(q+h q̇) - 8 J(q-h q̇) + J(q-2h q̇)] / (12 h)

    Then multiply by q̇ on the right.
    """
    q = asvec(q, np.asarray(q, float).size)
    qd = asvec(qd, q.size)
    h = float(eps)

    Jp2 = np.asarray(J_fn(q + 2.0 * h * qd), float)
    Jp1 = np.asarray(J_fn(q + 1.0 * h * qd), float)
    Jm1 = np.asarray(J_fn(q - 1.0 * h * qd), float)
    Jm2 = np.asarray(J_fn(q - 2.0 * h * qd), float)

    Jdot_dir = (-Jp2 + 8.0 * Jp1 - 8.0 * Jm1 + Jm2) / (12.0 * h)  # m×n
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
    # Accel blocks (body-frame)
    "S_from",
    "omega_from_Rdot",
    "alpha_from_Rddot",
    "classic_accel",
    # FD helper
    "jdot_qdot_fd",
]
