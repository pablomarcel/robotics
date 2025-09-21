# acceleration/tools/spatial.py
"""
Spatial-operator helpers for **acceleration kinematics**.

This module provides:
- 3×3 and 6×6 spatial algebra utilities:
    * tilde(v)                      → 3×3 skew
    * crossm(V) / crossf(V)         → 6×6 motion/force cross-product matrices
    * X_from_Rp(R, p)               → 6×6 motion transform X_AB
    * Xf_from_Rp(R, p)              → 6×6 force transform Xf_AB = (X_BA)^T
    * X_inv(X)                      → invert 6×6 motion transform
    * apply_X_to_twist(X, V)        → V_A = X_AB @ V_B
    * apply_Xf_to_wrench(Xf, F)     → F_A = Xf_AB @ F_B

- Point transport (classic rigid-body kinematics):
    * transport_velocity(omega, vB, r, v_rel=None)
    * transport_acceleration(alpha, omega, aB, r, v_rel=None, a_rel=None)
    * classic_accel(alpha, omega, r)  (thin re-export of utils.classic_accel)

- Mixed helper (representative of §9.4xx “mixed derivatives” style):
    * accel_point_from_frame(alpha, omega, r, aB=None, vB=None, a_rel=None, v_rel=None)

- Tiny utilities:
    * stack_twist(omega, v) / split_twist(V)
    * stack_wrench(n, f) / split_wrench(F)

Shapes
------
- 3-vectors: (3,)
- Twists/wrenches: (6,) stacked as [ω; v] and [n; f]
- 3×3 rotations, 6×6 transforms return float arrays
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..utils import skew as _skew, S_from as _S_from, classic_accel as _classic_accel


# -----------------------------------------------------------------------------
# Basic 3×3
# -----------------------------------------------------------------------------

def tilde(v: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    3×3 skew-symmetric matrix of a 3-vector.
    """
    return _skew(np.asarray(v, float).reshape(3))


# -----------------------------------------------------------------------------
# Spatial 6×6 cross-product operators
# -----------------------------------------------------------------------------

def crossm(V: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Motion cross-product matrix such that:
        crossm(V) @ W  ==  V × W            (spatial motion cross)
    where V = [ω; v], W = [ω2; v2].

    Returns
    -------
    (6,6) ndarray
        [[ ω~,  0 ],
         [ v~,  ω~]]
    """
    V = np.asarray(V, float).reshape(6)
    w = V[:3]
    v = V[3:]
    Wt = tilde(w)
    Vt = tilde(v)
    X = np.zeros((6, 6), float)
    X[:3, :3] = Wt
    X[3:, :3] = Vt
    X[3:, 3:] = Wt
    return X


def crossf(V: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Force cross-product matrix such that:
        crossf(V) @ F  ==  V ×* F           (spatial force cross)
    where F = [n; f] is a wrench.

    Returns
    -------
    (6,6) ndarray
        [[ -ω~, -v~ ],
         [  0 , -ω~ ]]
    """
    V = np.asarray(V, float).reshape(6)
    w = V[:3]
    v = V[3:]
    Wt = tilde(w)
    Vt = tilde(v)
    X = np.zeros((6, 6), float)
    X[:3, :3] = -Wt
    X[:3, 3:] = -Vt
    X[3:, 3:] = -Wt
    return X


# -----------------------------------------------------------------------------
# Spatial 6×6 transforms
# -----------------------------------------------------------------------------

def X_from_Rp(R: np.ndarray, p: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Motion transform X_AB given rotation R_AB and translation p_AB (A←B).

    X_AB maps a twist expressed in B to a twist expressed in A:
        V_A = X_AB @ V_B

    Returns
    -------
    (6,6) ndarray
        [[ R,  0 ],
         [ p^R, R ]]
    """
    R = np.asarray(R, float).reshape(3, 3)
    p = np.asarray(p, float).reshape(3)
    X = np.zeros((6, 6), float)
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = tilde(p) @ R
    return X


def Xf_from_Rp(R: np.ndarray, p: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Force transform Xf_AB corresponding to X_AB:
        F_A = Xf_AB @ F_B

    Relationship:
        Xf_AB = (X_BA)^T
    """
    R = np.asarray(R, float).reshape(3, 3)
    p = np.asarray(p, float).reshape(3)
    Xf = np.zeros((6, 6), float)
    Xf[:3, :3] = R
    Xf[:3, 3:] = tilde(p) @ R
    Xf[3:, 3:] = R
    return Xf


def X_inv(X: np.ndarray) -> np.ndarray:
    """
    Invert a motion transform X = X_AB to obtain X_BA.

    For X = [[R, 0], [p^R, R]] we have:
        X^{-1} = [[R^T, 0], [-R^T p^, R^T]]
    """
    X = np.asarray(X, float).reshape(6, 6)
    R = X[:3, :3]
    RT = R.T
    # p^R = skew(p) @ R ⇒ get p^ as (p^R) @ R^T
    pR = X[3:, :3]
    p_hat = pR @ RT
    Xinv = np.zeros((6, 6), float)
    Xinv[:3, :3] = RT
    Xinv[3:, 3:] = RT
    Xinv[3:, :3] = -RT @ p_hat
    return Xinv


def apply_X_to_twist(X: np.ndarray, V: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Apply motion transform to a twist: V' = X @ V.
    """
    X = np.asarray(X, float).reshape(6, 6)
    V = np.asarray(V, float).reshape(6)
    return X @ V


def apply_Xf_to_wrench(Xf: np.ndarray, F: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Apply force transform to a wrench: F' = Xf @ F.
    """
    Xf = np.asarray(Xf, float).reshape(6, 6)
    F = np.asarray(F, float).reshape(6)
    return Xf @ F


# -----------------------------------------------------------------------------
# Twists & wrenches convenience
# -----------------------------------------------------------------------------

def stack_twist(omega: Sequence[float] | np.ndarray, v: Sequence[float] | np.ndarray) -> np.ndarray:
    """Stack 3+3 → 6 twist [ω; v]."""
    return np.r_[np.asarray(omega, float).reshape(3), np.asarray(v, float).reshape(3)]


def split_twist(V: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split 6 twist → (ω, v)."""
    V = np.asarray(V, float).reshape(6)
    return V[:3].copy(), V[3:].copy()


def stack_wrench(n: Sequence[float] | np.ndarray, f: Sequence[float] | np.ndarray) -> np.ndarray:
    """Stack 3+3 → 6 wrench [n; f]."""
    return np.r_[np.asarray(n, float).reshape(3), np.asarray(f, float).reshape(3)]


def split_wrench(F: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split 6 wrench → (n, f)."""
    F = np.asarray(F, float).reshape(6)
    return F[:3].copy(), F[3:].copy()


# -----------------------------------------------------------------------------
# Point transport (velocity & acceleration)
# -----------------------------------------------------------------------------

def transport_velocity(omega: Sequence[float] | np.ndarray,
                       vB: Sequence[float] | np.ndarray,
                       r: Sequence[float] | np.ndarray,
                       v_rel: Optional[Sequence[float] | np.ndarray] = None) -> np.ndarray:
    """
    Velocity transport from a moving frame origin B to a point P=B+r:

        v_P = v_B + ω × r + v_rel

    Set v_rel=None for a fixed point in the body.
    """
    ω = np.asarray(omega, float).reshape(3)
    vB = np.asarray(vB, float).reshape(3)
    r = np.asarray(r, float).reshape(3)
    vrel = np.zeros(3) if v_rel is None else np.asarray(v_rel, float).reshape(3)
    return vB + np.cross(ω, r) + vrel


def transport_acceleration(alpha: Sequence[float] | np.ndarray,
                           omega: Sequence[float] | np.ndarray,
                           aB: Sequence[float] | np.ndarray,
                           r: Sequence[float] | np.ndarray,
                           v_rel: Optional[Sequence[float] | np.ndarray] = None,
                           a_rel: Optional[Sequence[float] | np.ndarray] = None) -> np.ndarray:
    """
    Acceleration transport from a moving frame origin B to a point P=B+r:

        a_P = a_B + α × r + ω × (ω × r) + 2 ω × v_rel + a_rel

    Set v_rel=a_rel=0 for a *fixed* point in the body (classic result).
    """
    α = np.asarray(alpha, float).reshape(3)
    ω = np.asarray(omega, float).reshape(3)
    aB = np.asarray(aB, float).reshape(3)
    r = np.asarray(r, float).reshape(3)
    vrel = np.zeros(3) if v_rel is None else np.asarray(v_rel, float).reshape(3)
    arel = np.zeros(3) if a_rel is None else np.asarray(a_rel, float).reshape(3)
    return aB + np.cross(α, r) + np.cross(ω, np.cross(ω, r)) + 2.0 * np.cross(ω, vrel) + arel


def classic_accel(alpha: Sequence[float], omega: Sequence[float], r: Sequence[float]) -> np.ndarray:
    """
    Thin wrapper to the classic rigid-body point acceleration:
        a = α×r + ω×(ω×r)
    """
    return _classic_accel(alpha, omega, r)


# -----------------------------------------------------------------------------
# Mixed convenience (representative of §9.4xx cases)
# -----------------------------------------------------------------------------

def accel_point_from_frame(alpha: Sequence[float] | np.ndarray,
                           omega: Sequence[float] | np.ndarray,
                           r: Sequence[float] | np.ndarray,
                           aB: Optional[Sequence[float] | np.ndarray] = None,
                           vB: Optional[Sequence[float] | np.ndarray] = None,
                           a_rel: Optional[Sequence[float] | np.ndarray] = None,
                           v_rel: Optional[Sequence[float] | np.ndarray] = None) -> np.ndarray:
    """
    Compute the acceleration of a point P at offset r from a moving frame origin B.

    Parameters
    ----------
    alpha, omega : (3,)
        Angular acceleration and velocity of the frame B.
    r : (3,)
        Vector from B-origin to the point P, expressed in the same frame.
    aB, vB : (3,), optional
        Linear acceleration and velocity of the B-origin. Defaults to zeros.
    a_rel, v_rel : (3,), optional
        Relative linear acceleration/velocity of P w.r.t the body (for sliding
        points, etc.). Defaults to zeros.

    Returns
    -------
    aP : (3,)
        Acceleration of the point P.

    Notes
    -----
    This folds the classic transport terms:
        a_P = a_B + α×r + ω×(ω×r) + 2 ω× v_rel + a_rel

    If you pass v_rel=a_rel=None it reduces to the classic fixed-point formula.
    """
    aB = np.zeros(3) if aB is None else np.asarray(aB, float).reshape(3)
    vB = np.zeros(3) if vB is None else np.asarray(vB, float).reshape(3)
    a_fixed = transport_acceleration(alpha, omega, aB, r, v_rel=np.zeros(3), a_rel=np.zeros(3))
    extra = np.zeros(3)
    if v_rel is not None:
        extra += 2.0 * np.cross(np.asarray(omega, float).reshape(3), np.asarray(v_rel, float).reshape(3))
    if a_rel is not None:
        extra += np.asarray(a_rel, float).reshape(3)
    return a_fixed + extra


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    # 3×3
    "tilde",
    # spatial 6×6 ops
    "crossm",
    "crossf",
    "X_from_Rp",
    "Xf_from_Rp",
    "X_inv",
    "apply_X_to_twist",
    "apply_Xf_to_wrench",
    # twists/wrenches
    "stack_twist",
    "split_twist",
    "stack_wrench",
    "split_wrench",
    # transport / classic
    "transport_velocity",
    "transport_acceleration",
    "classic_accel",
    # mixed convenience
    "accel_point_from_frame",
]
