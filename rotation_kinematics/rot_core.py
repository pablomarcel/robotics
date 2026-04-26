# =============================
# File: rotation_kinematics/rot_core.py
# =============================
from __future__ import annotations
from typing import Dict, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from .rot_utils import (
    VALID_ALL, VALID_TB, VALID_PROPER,   # kept for external use if others import from here
    compose_global, compose_local, decompose,
    rotation_power, align_x_to_vector,
    is_rotmat, save_matrix_csv, save_series_csv,
)

# ---------------------------------------------------------------------
# Core builders / validators
# ---------------------------------------------------------------------

def build_matrix(mode: str, seq: str, angles: Sequence[float], degrees: bool = False) -> R:
    """
    Build a scipy Rotation for a given convention ('global'|'local'), sequence, and angles.
    Angles are interpreted in degrees if degrees=True, otherwise radians.
    """
    mode = mode.lower()
    if mode not in {"global", "local"}:
        raise ValueError("mode must be 'global' or 'local'")
    if mode == 'global':
        return compose_global(seq, angles, degrees=degrees)
    else:
        return compose_local(seq, angles, degrees=degrees)


def transform_points(Robj: R, pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Active transform of row-stacked points/vectors using Rotation.apply."""
    P = np.asarray(pts, float)
    return Robj.apply(P)


def passive_transform(Robj: R, vecs: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Passive change of coordinates (global -> body): multiply by R^T.
    Implemented via Rotation.inv().apply on row-stacked vectors.
    """
    return Robj.inv().apply(np.asarray(vecs, float))


def check_matrix(Robj: R) -> Dict:
    """Return orthogonality diagnostics for a Rotation."""
    ok, info = is_rotmat(Robj.as_matrix())
    info["ok"] = ok
    return info


# ---------------------------------------------------------------------
# Euler / RPY: ω  <->  q̇  (fixed & body frames, any proper sequence)
# ---------------------------------------------------------------------

def _vee(S: np.ndarray) -> np.ndarray:
    """vee: so(3)->R^3 for a (nearly) skew-symmetric matrix S."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def _rotation_matrix(seq: str, q_rad: Sequence[float], convention: str = 'global') -> np.ndarray:
    """
    Build a 3x3 rotation_kinematics matrix for given sequence and angles (radians).
    convention: 'global' (extrinsic) or 'local' (intrinsic).
    """
    convention = convention.lower()
    if convention not in {"global", "local"}:
        raise ValueError("convention must be 'global' or 'local'")
    if convention == 'global':
        return compose_global(seq, q_rad, degrees=False).as_matrix()
    else:
        return compose_local(seq, q_rad, degrees=False).as_matrix()


def _omega_jacobian_numeric(
    seq: str,
    q_rad: np.ndarray,
    convention: str = 'global',
    frame: str = 'space',
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Build numeric Jacobian M(q) so that omega = M(q) @ qdot.
    - q_rad in radians
    - frame: 'space' (ω in global) uses Ω = Ṙ Rᵀ ; 'body' (ω in body) uses Ω = Rᵀ Ṙ
    Central difference is used for robustness and correctness.
    """
    frame = frame.lower()
    if frame not in {"space", "body"}:
        raise ValueError("frame must be 'space' or 'body'")

    R0 = _rotation_matrix(seq, q_rad, convention)
    cols = []

    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        Rp = _rotation_matrix(seq, q_rad + dq, convention)
        Rm = _rotation_matrix(seq, q_rad - dq, convention)
        Rdot = (Rp - Rm) / (2.0 * eps)

        if frame == 'space':
            Omega = Rdot @ R0.T
        else:  # 'body'
            Omega = R0.T @ Rdot

        # enforce skew to suppress numeric noise
        Omega = 0.5 * (Omega - Omega.T)
        cols.append(_vee(Omega))

    return np.column_stack(cols)  # 3x3


def angvel_from_rates(
    seq: str,
    angles: Sequence[float],
    rates: Sequence[float],
    convention: str = 'global',
    degrees: bool = False,
    frame: str = 'body',
) -> np.ndarray:
    """
    Map generalized angle rates q̇ -> angular_velocity velocity_kinematics ω.

    Parameters
    ----------
    seq : str
        Proper Euler / Tait-Bryan sequence (e.g., 'zyx', 'zyz', 'xyz', ...).
    angles : (3,)
        Angles (deg if degrees=True, else rad).
    rates : (3,)
        Angle rates q̇ (deg/s if degrees=True, else rad/s).
    convention : {'global','local'}
        'global' for extrinsic rotations, 'local' for intrinsic rotations.
    degrees : bool
        Interpret angles and rates in degrees/deg/s when True.
    frame : {'space','body'}
        Frame of ω. 'space' returns ω in global frame; 'body' in body frame.

    Returns
    -------
    ω : (3,)
        Angular velocity_kinematics (deg/s if degrees=True, else rad/s).
    """
    q = np.deg2rad(angles) if degrees else np.asarray(angles, float)
    qdot = np.deg2rad(rates) if degrees else np.asarray(rates, float)

    M = _omega_jacobian_numeric(seq, q, convention=convention, frame=frame)
    omega = M @ qdot

    return np.rad2deg(omega) if degrees else omega


def rates_from_angvel(
    seq: str,
    angles: Sequence[float],
    omega: Sequence[float],
    convention: str = 'global',
    degrees: bool = False,
    frame: str = 'body',
    rcond: float = 1e-12,
) -> np.ndarray:
    """
    Map angular_velocity velocity_kinematics ω -> generalized rates q̇ using a pseudoinverse.

    Parameters
    ----------
    omega : (3,)
        Angular velocity_kinematics (deg/s if degrees=True, else rad/s).
    rcond : float
        Cutoff for small singular values in the pseudoinverse; helps near singularities.

    Returns
    -------
    q̇ : (3,)
        Angle rates (deg/s if degrees=True, else rad/s).
    """
    q = np.deg2rad(angles) if degrees else np.asarray(angles, float)
    w = np.deg2rad(omega) if degrees else np.asarray(omega, float)

    M = _omega_jacobian_numeric(seq, q, convention=convention, frame=frame)
    qdot = np.linalg.pinv(M, rcond=rcond) @ w

    return np.rad2deg(qdot) if degrees else qdot


# ---------------------------------------------------------------------
# Equations 2.84/2.85 style repeat (power of a rotation_kinematics)
# ---------------------------------------------------------------------

def repeat_rotation(Robj: R, m: int) -> R:
    """Return R^m using the logarithm/exponential map (stable for integer m)."""
    return rotation_power(Robj, m)


# ---------------------------------------------------------------------
# Alignment (2.99–2.106): align body x-axis with a given vector u
# ---------------------------------------------------------------------

def align_body_x(u: Sequence[float]) -> R:
    """Return a Rotation that aligns the body x-axis with vector u (right-handed)."""
    return align_x_to_vector(u)


# ---------------------------------------------------------------------
# Convenience wrappers for CSV I/O
# ---------------------------------------------------------------------

def save_R(path: str, Robj: R) -> None:
    """Save a rotation_kinematics matrix to CSV at rotation_kinematics/out/<path_planning>."""
    save_matrix_csv(path, Robj)


def save_series(path: str, t: np.ndarray, data: np.ndarray, header: str) -> None:
    """Save a time series (t, data[*,3]) to CSV at rotation_kinematics/out/<path_planning> with a header."""
    save_series_csv(path, t, data, header)
