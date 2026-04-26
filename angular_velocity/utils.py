from __future__ import annotations
import functools
import numpy as np
from typing import Callable, Tuple

Array = np.ndarray

def skew(omega: Array) -> Array:
    """Return the 3x3 skew-symmetric matrix (hat) of a 3-vector."""
    wx, wy, wz = omega.reshape(3)
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=float)

def vee(Omega: Array) -> Array:
    """Inverse of skew (vee) → 3-vector from 3x3 skew-symmetric matrix."""
    return np.array([Omega[2,1], Omega[0,2], Omega[1,0]], dtype=float)

def is_rotation_matrix(R: Array, atol: float = 1e-8) -> bool:
    return R.shape == (3,3) and np.allclose(R @ R.T, np.eye(3), atol=atol) and np.isclose(np.linalg.det(R), 1.0, atol=atol)

def ensure_rotation(func: Callable) -> Callable:
    """Decorator: validate rotation_kinematics matrix args named 'R' or attributes 'self.R'."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # try to find R from args/kwargs/self
        R = kwargs.get("R", None)
        if R is None and len(args) >= 1 and hasattr(args[0], "R"):
            R = getattr(args[0], "R")
        if R is not None and not is_rotation_matrix(np.asarray(R)):
            raise ValueError("Input is not a proper rotation_kinematics matrix.")
        return func(*args, **kwargs)
    return wrapper

def block44(R: Array, d: Array) -> Array:
    """Build 4x4 homogeneous transform from R (3x3) and d (3,)."""
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = d.reshape(3)
    return T

def adjoint(T: Array) -> Array:
    """Adjoint of SE(3) transform T (4x4) → 6x6."""
    R = T[:3,:3]; p = T[:3,3]
    Ad = np.zeros((6,6))
    Ad[:3,:3] = R
    Ad[3:,3:] = R
    Ad[3:,:3] = skew(p) @ R
    return Ad

def motion_cross(V: Array) -> Array:
    """Small-adjoint (motion_kinematics cross product matrix) ad_V, 6x6."""
    w = V[:3]; v = V[3:]
    ad = np.zeros((6,6))
    ad[:3,:3] = skew(w)
    ad[3:,3:] = skew(w)
    ad[3:,:3] = skew(v)
    return ad

def save_array(path: str, arr: Array) -> None:
    np.save(path, arr)

def load_array(path: str) -> Array:
    return np.load(path)
