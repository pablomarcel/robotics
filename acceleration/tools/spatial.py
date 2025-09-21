# acceleration/tools/spatial.py
"""
Spatial-operator helpers for **acceleration kinematics**.

This module provides:
- 3×3 and 6×6 spatial algebra utilities:
    * skew(v) / tilde(v)            → 3×3 skew (hat) of a vector
    * vex(S)                        → vee operator
    * adjoint(T)                    → 6×6 Ad_T from SE(3)
    * X_from_Rp(R, p)               → 6×6 motion transform X_AB
    * Xf_from_Rp(R, p)              → 6×6 force transform Xf_AB = (X_BA)^T
    * motion_xform(T)               → X_AB from 4×4 SE(3) (convenience)
    * force_xform(T)                → Xf_AB from 4×4 SE(3) (convenience)
    * cross_motion(V) / cross_force(V)
    * spatial_inertia(m, com, Ic)   → 6×6 spatial inertia about frame origin

- Point transport (classic rigid-body kinematics):
    * transport_velocity(omega, vB, r, v_rel=None)
    * transport_acceleration(alpha, omega, aB, r, v_rel=None, a_rel=None)
    * classic_accel(alpha, omega, r)  (thin re-export of utils.classic_accel)

- Mixed helper:
    * accel_point_from_frame(alpha, omega, r, aB=None, vB=None, a_rel=None, v_rel=None)

Shapes
------
- 3-vectors: (3,)
- Twists/wrenches: (6,) stacked as [ω; v] and [n; f]
- 3×3 rotations, 6×6 transforms return float arrays
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..utils import (
    skew as _skew,
    vex as _vex,
    adjoint as _adjoint,
    classic_accel as _classic_accel,
)

# -----------------------------------------------------------------------------
# Basic 3×3 (hat / vee)
# -----------------------------------------------------------------------------

def tilde(v: Sequence[float] | np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix of a 3-vector (a.k.a. hat operator)."""
    return _skew(np.asarray(v, float).reshape(3))


def skew(v: Sequence[float] | np.ndarray) -> np.ndarray:
    """Alias of :func:`tilde` for convenience (many tests import `skew`)."""
    return tilde(v)


def vex(S: np.ndarray) -> np.ndarray:
    """Vee operator re-export (inverse of skew)."""
    return _vex(S)


# -----------------------------------------------------------------------------
# Adjoint (SE(3) → R^{6×6})
# -----------------------------------------------------------------------------

def adjoint(T: np.ndarray) -> np.ndarray:
    """Re-export Ad_T from utils for convenience in spatial tests."""
    return _adjoint(T)


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


# Friendly aliases to match test names
def cross_motion(V: Sequence[float] | np.ndarray) -> np.ndarray:
    """Alias of :func:`crossm`."""
    return crossm(V)


def cross_force(V: Sequence[float] | np.ndarray) -> np.ndarray:
    """Alias of :func:`crossf`."""
    return crossf(V)


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
    # p^R = skew(p) @ R ⇒ recover p^ as (p^R) @ R^T
    pR = X[3:, :3]
    p_hat = pR @ RT
    Xinv = np.zeros((6, 6), float)
    Xinv[:3, :3] = RT
    Xinv[3:, 3:] = RT
    Xinv[3:, :3] = -RT @ p_hat
    return Xinv


def apply_X_to_twist(X: np.ndarray, V: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply motion transform to a twist: V' = X @ V."""
    X = np.asarray(X, float).reshape(6, 6)
    V = np.asarray(V, float).reshape(6)
    return X @ V


def apply_Xf_to_wrench(Xf: np.ndarray, F: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply force transform to a wrench: F' = Xf @ F."""
    Xf = np.asarray(Xf, float).reshape(6, 6)
    F = np.asarray(F, float).reshape(6)
    return Xf @ F


# Convenience wrappers that accept a 4×4 SE(3) transform (as tests expect)
def motion_xform(T: np.ndarray) -> np.ndarray:
    """X_AB from 4×4 SE(3) T_AB."""
    T = np.asarray(T, float).reshape(4, 4)
    return X_from_Rp(T[:3, :3], T[:3, 3])


def force_xform(T: np.ndarray) -> np.ndarray:
    """Xf_AB from 4×4 SE(3) T_AB."""
    T = np.asarray(T, float).reshape(4, 4)
    return Xf_from_Rp(T[:3, :3], T[:3, 3])


# -----------------------------------------------------------------------------
# Spatial inertia
# -----------------------------------------------------------------------------

def spatial_inertia(m: float, com: Sequence[float] | np.ndarray, Ic: np.ndarray) -> np.ndarray:
    """
    Spatial inertia about the *frame origin* given:
      - m  : mass
      - com: CoM vector expressed in the body frame (3,)
      - Ic : rotational inertia about the CoM (3×3, symmetric positive definite)

    Block form:
        I = [[Ic + m [c] [c]^T,   m [c]],
             [ -m [c],            m I3]]
    """
    m = float(m)
    c = np.asarray(com, float).reshape(3)
    Ic = np.asarray(Ic, float).reshape(3, 3)
    c_hat = skew(c)

    I = np.zeros((6, 6), float)
    I[:3, :3] = Ic + m * (c_hat @ c_hat.T)
    I[:3, 3:] = m * c_hat
    I[3:, :3] = -m * c_hat
    I[3:, 3:] = m * np.eye(3)
    return I


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
# Mixed convenience
# -----------------------------------------------------------------------------

def accel_point_from_frame(alpha: Sequence[float] | np.ndarray,
                           omega: Sequence[float] | np.ndarray,
                           r: Sequence[float] | np.ndarray,
                           aB: Optional[Sequence[float] | np.ndarray] = None,
                           vB: Optional[Sequence[float] | np.ndarray] = None,
                           a_rel: Optional[Sequence[float] | np.ndarray] = None,
                           v_rel: Optional[Sequence[float] | np.ndarray] = None) -> np.ndarray:
    """
    Acceleration of a point P at offset r from a moving frame origin B:

        a_P = a_B + α×r + ω×(ω×r) + 2 ω× v_rel + a_rel

    If v_rel=a_rel=None this reduces to the classic fixed-point formula.
    """
    aB = np.zeros(3) if aB is None else np.asarray(aB, float).reshape(3)
    _ = vB  # unused; kept for signature symmetry with velocity transport
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
    "skew",
    "vex",
    # adjoint
    "adjoint",
    # spatial 6×6 ops
    "crossm",
    "crossf",
    "cross_motion",
    "cross_force",
    "X_from_Rp",
    "Xf_from_Rp",
    "X_inv",
    "apply_X_to_twist",
    "apply_Xf_to_wrench",
    "motion_xform",
    "force_xform",
    # spatial inertia
    "spatial_inertia",
    # transport / classic
    "transport_velocity",
    "transport_acceleration",
    "classic_accel",
    # mixed convenience
    "accel_point_from_frame",
]
