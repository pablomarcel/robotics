# inverse/utils.py
"""
Utility decorators and math helpers for **inverse kinematics**.

This module mirrors the forward toolkit’s style but adds IK-oriented helpers:
- Decorators:
    * timed(fn)            → records fn runtime (seconds) on wrapper.last_runtime
    * ensure_shape(*shape) → validates ndarray shape of the return value

- Numerical helpers:
    * TOL                           (default tolerance)
    * almost_equal(a, b)
    * clamp(x, lo, hi)
    * normalize(v)

- Lie/SE(3) helpers:
    * skew(v) / vex(S)              (hat / vee)
    * hat_xi(ω, v)                  → 4×4 se(3) from screw
    * adjoint(T)                    → 6×6 Ad_T
    * homogeneous(R, t)             → 4×4 from (R, t)
    * so3_log(R)                    → rotation vector (axis-angle in R^3)
    * se3_log(T)                    → 6×1 twist (approximate; good for small motions)

- IK helpers:
    * rpy_to_R(roll, pitch, yaw)    → Rz(yaw) Ry(pitch) Rx(roll)
    * R_to_rpy(R)
    * pose_error(T_curr, T_des, *, mode="small")  → 6×1 task-space error
    * dls_step(J, e, lam)           → damped least-squares Δq
    * manipulability_metrics(J)     → {min_sing, cond, detJJT, rank}

These are NumPy-only to keep CI light and tests fast.

References to Chapter 6
-----------------------
- Geometric Jacobian & conditioning: (6.289–6.296), (singularity discussion 6.324–6.329)
- Newton/LM IK step: (6.298–6.299)
- Small-angle SO(3) log used for orientation error in iterative IK.
"""
from __future__ import annotations

import functools
import math
import time
from typing import Callable, Dict, Optional, Sequence, Tuple, TypeVar

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


def homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compose a 4×4 homogeneous transform from (R, t).
    """
    H = np.eye(4, dtype=float)
    H[:3, :3] = np.asarray(R, float).reshape(3, 3)
    H[:3, 3] = np.asarray(t, float).reshape(3)
    return H


# ------------------------------ SO(3)/SE(3) log ---------------------------

def so3_log(R: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Map SO(3) → R^3 (rotation vector). Returns ω such that exp([ω]^)=R.

    For small angles, this reduces to vex(R - R^T)/2 (small-angle approx).
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_phi = (np.trace(R) - 1.0) * 0.5
    cos_phi = float(np.clip(cos_phi, -1.0, 1.0))
    phi = math.acos(cos_phi)
    if phi < 1e-8:
        # Small-angle: ω ≈ vee(R - R^T)/2
        return vex(R - R.T)
    # General case
    omega_hat = (R - R.T) * (0.5 / math.sin(phi))
    u = vex(omega_hat)
    return u * phi


def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Approximate log map SE(3) → R^6; returns [ω; v] such that exp([ξ]^)=T.
    Uses a first-order approximation appropriate for small motions and for
    iterative IK error vectors (δ pose). For larger motions, use a full SO(3)
    log and the series for G^{-1}; that’s overkill for per-iteration errors.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    w = so3_log(R)
    return np.r_[w, p]


# ---------------------------------------------------------------------------
# IK-oriented helpers
# ---------------------------------------------------------------------------

def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Construct rotation R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def R_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Inverse of rpy_to_R (yaw-pitch-roll convention).
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    sy = -R[2, 0]
    cy = math.sqrt(max(0.0, 1.0 - sy * sy))
    if cy < 1e-9:
        pitch = math.asin(sy)
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        pitch = math.asin(sy)
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def pose_error(
    T_curr: np.ndarray,
    T_des: np.ndarray,
    *,
    mode: str = "small",
) -> np.ndarray:
    """
    Compute a 6×1 task-space error e = [dp; dθ] between two poses.

    Parameters
    ----------
    T_curr, T_des : (4,4) ndarray
        Current and desired transforms.
    mode : {"small", "so3"}
        - "small": uses small-angle rotation vector: 0.5 * vee(Rerr - Rerr^T)
        - "so3":   uses full SO(3) log for orientation

    Returns
    -------
    (6,) ndarray
        e = [dx, dy, dz, wx, wy, wz]
    """
    T_curr = np.asarray(T_curr, float).reshape(4, 4)
    T_des = np.asarray(T_des, float).reshape(4, 4)
    dp = T_des[:3, 3] - T_curr[:3, 3]
    Rerr = T_curr[:3, :3].T @ T_des[:3, :3]
    if mode == "so3":
        w = so3_log(Rerr)
    else:  # "small"
        w = 0.5 * vex(Rerr - Rerr.T)
    return np.r_[dp, w]


def dls_step(J: np.ndarray, e: np.ndarray, lam: float) -> np.ndarray:
    """
    Damped least-squares (Levenberg–Marquardt) step:
        Δq = (JᵀJ + λ² I)⁻¹ Jᵀ e
    Matches (6.298–6.299). Handles over/under/fully determined cases.
    """
    J = np.asarray(J, float)
    e = np.asarray(e, float).reshape(-1)
    n = J.shape[1]
    JTJ = J.T @ J
    return np.linalg.solve(JTJ + (lam ** 2) * np.eye(n), J.T @ e)


def manipulability_metrics(J: np.ndarray, *, tol: float = 1e-8) -> Dict[str, float]:
    """
    Basic singularity metrics for a geometric Jacobian (6.324–6.329).
    Returns
    -------
    dict with keys:
      - min_sing : smallest singular value σ_min
      - cond     : σ_max / σ_min (∞ if σ_min≈0)
      - detJJT   : determinant of J Jᵀ (product σ_i^2 for square J)
      - rank     : numeric rank
    """
    J = np.asarray(J, float)
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    sigma_min = float(S[-1]) if S.size else 0.0
    sigma_max = float(S[0]) if S.size else 0.0
    cond = float(np.inf if sigma_min < tol else sigma_max / sigma_min)
    detJJT = float(np.prod(S) ** 2) if S.size else 0.0
    rank = int((S > tol).sum())
    return {"min_sing": sigma_min, "cond": cond, "detJJT": detJJT, "rank": rank}
