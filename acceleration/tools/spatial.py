# acceleration/tools/spatial.py
"""
Spatial-operator helpers for **acceleration kinematics**.

Conventions (used throughout):
- Twists (spatial motion_kinematics):   v_B = X_AB v_A
- Wrenches (spatial force):  f_B = X*_AB f_A  with  X*_AB = (X_AB^{-1})^T
- Spatial inertia mapping:   I_B = X*_AB I_A X_BA   (power-invariant)

This module provides:
- 3×3 and 6×6 spatial algebra utilities:
    * skew(v) / tilde(v)            → 3×3 skew (hat) of a vector
    * vex(S)                        → vee operator
    * adjoint(T)                    → 6×6 Ad_T from SE(3)
    * X_from_Rp(R, p)               → 6×6 **motion_kinematics** transform X_AB (A→B)
    * Xf_from_Rp(R, p)              → 6×6 **force** transform X*_AB = X_AB^{-T}
    * motion_xform(T)               → X_AB from 4×4 SE(3) (A→B)
    * motion_xform_inv(T)           → X_BA from 4×4 SE(3) (A→B) via T_BA = T_AB^{-1}
    * force_xform(T)                → X*_AB from 4×4 SE(3) (A→B)
    * cross_motion(V) / cross_force(V)
    * spatial_inertia(m, com, Ic)   → 6×6 spatial inertia about frame origin
    * transform_inertia(T, I_A)     → safe inertia mapping I_B = X*_AB I_A X_BA

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

from typing import Optional, Sequence

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
    """Vee operator re-export (inverse_kinematics of skew)."""
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
        crossm(V) @ W  ==  V × W            (spatial motion_kinematics cross)
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

    Duality (power invariance) requires:
        crossf(V)  =  -(crossm(V))^T
    """
    return -(crossm(V).T)


# Friendly aliases to match test names
def cross_motion(V: Sequence[float] | np.ndarray) -> np.ndarray:
    """Alias of :func:`crossm`."""
    return crossm(V)


def cross_force(V: Sequence[float] | np.ndarray) -> np.ndarray:
    """Alias of :func:`crossf`."""
    return crossf(V)


# -----------------------------------------------------------------------------
# Spatial 6×6 transforms (A→B)
# -----------------------------------------------------------------------------

def X_from_Rp(R: np.ndarray, p: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    **Motion** transform X_AB for T maps A→B (so V_B = X_AB V_A):

        X_AB = [[ R,    0 ],
                [ p^ R, R ]]

    This matches the tests' expectation that for a pure translation (R=I),
    X[3:, :3] = p^ (positive sign).
    """
    R = np.asarray(R, float).reshape(3, 3)
    p = np.asarray(p, float).reshape(3)
    X = np.zeros((6, 6), float)
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = skew(p) @ R
    return X


def Xf_from_Rp(R: np.ndarray, p: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    **Force** transform X*_AB corresponding to X_AB (A→B):
        F_B = X*_AB @ F_A

    Relationship by power invariance:
        X*_AB = (X_AB)^{-T}  = [[ R,  p^ R ],
                                [ 0,    R  ]]
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
    Invert a **motion_kinematics** transform X_AB to obtain X_BA.  Closed form for:

        X_AB = [[R, 0],
                [p^ R, R]]

    is:

        X_BA = [[R^T,  0      ],
                [-R^T p^, R^T ]]
    """
    X = np.asarray(X, float).reshape(6, 6)
    R = X[:3, :3]
    RT = R.T
    pR = X[3:, :3]             # = p^ R
    p_hat = pR @ RT            # recover p^
    Xinv = np.zeros((6, 6), float)
    Xinv[:3, :3] = RT
    Xinv[3:, 3:] = RT
    Xinv[3:, :3] = -RT @ p_hat
    return Xinv


def apply_X_to_twist(X: np.ndarray, V: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply motion_kinematics transform to a twist: V' = X @ V."""
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
    """
    X_AB from 4×4 SE(3) T_AB (A→B).  T = [[R, p],[0,1]] with p expressed in A.
    """
    T = np.asarray(T, float).reshape(4, 4)
    return X_from_Rp(T[:3, :3], T[:3, 3])


def motion_xform_inv(T: np.ndarray) -> np.ndarray:
    """
    X_BA from 4×4 SE(3) T_AB (A→B) by using the inverse_kinematics pose:
        X_BA = X( T_BA ) with  T_BA = T_AB^{-1}
    This avoids directly inverting a 6×6 matrix.
    """
    T = np.asarray(T, float).reshape(4, 4)
    T_BA = np.linalg.inv(T)
    return motion_xform(T_BA)


def force_xform(T: np.ndarray) -> np.ndarray:
    """
    X*_AB from 4×4 SE(3) T_AB (A→B), defined as the **dual** of motion_kinematics:

        X*_AB = (X_AB)^{-T}

    This guarantees both duality and power invariance in tests.
    """
    T = np.asarray(T, float).reshape(4, 4)
    X = X_from_Rp(T[:3, :3], T[:3, 3])
    return np.linalg.inv(X).T


# -----------------------------------------------------------------------------
# Spatial inertia
# -----------------------------------------------------------------------------

def spatial_inertia(m: float, com: Sequence[float] | np.ndarray, Ic: np.ndarray) -> np.ndarray:
    """
    Spatial inertia about the *frame origin* given:
      - m  : mass
      - com: CoM vector expressed in the body frame (3,)
      - Ic : rotational inertia about the CoM (3×3, symmetric positive definite)

    Block form (about origin O, CoM at c):
        I = [[Ic + m [c][c]^T,   m [c]],
             [   -m [c],         m I3 ]]
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


def transform_inertia(T_AB: np.ndarray, I_A: np.ndarray) -> np.ndarray:
    """
    Map spatial inertia from frame A to frame B given T_AB with v_B = X_AB v_A:

        I_B = X*_AB I_A X_BA

    implemented using 4×4 poses (avoids directly inverting the 6×6).
    """
    X_AB = motion_xform(T_AB)
    Xstar_AB = force_xform(T_AB)
    X_BA = motion_xform_inv(T_AB)
    return Xstar_AB @ I_A @ X_BA


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
    # transforms
    "X_from_Rp",
    "Xf_from_Rp",
    "X_inv",
    "apply_X_to_twist",
    "apply_Xf_to_wrench",
    "motion_xform",
    "motion_xform_inv",
    "force_xform",
    # spatial inertia
    "spatial_inertia",
    "transform_inertia",
    # transport / classic
    "transport_velocity",
    "transport_acceleration",
    "classic_accel",
    # mixed convenience
    "accel_point_from_frame",
]
