# =============================
# File: rotation/rot_utils.py
# =============================
from __future__ import annotations

import math
import os
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

# SciPy Rotation is the numerical workhorse for all rotation matrices
try:
    from scipy.spatial.transform import Rotation as _R
except Exception as _e:  # pragma: no cover
    raise RuntimeError("SciPy is required: pip install scipy") from _e

R = _R  # alias for external modules

# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def ensure_dir(p: str) -> None:
    """Create parent directories for a path if they don't exist."""
    if p == "":
        return
    os.makedirs(p, exist_ok=True)


def parse_floats(csv: str | Sequence[float]) -> List[float]:
    """Parse '1,2,3' or ['1','2','3'] into [1.0, 2.0, 3.0]."""
    if isinstance(csv, (list, tuple, np.ndarray)):
        return [float(x) for x in csv]
    parts = [s for s in str(csv).replace(" ", "").split(",") if s != ""]
    return [float(x) for x in parts]


def parse_bool(x: str | bool | None, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def hat(v: ArrayLike) -> np.ndarray:
    """so(3) hat operator: R^3 -> skew(3)."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]],
        dtype=float,
    )


def vee(M: ArrayLike) -> np.ndarray:
    """vee operator: skew(3) -> R^3."""
    M = np.asarray(M, dtype=float)
    return np.array(
        [M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]],
        dtype=float,
    ) * 0.5


def is_rotmat(Rm: ArrayLike, tol: float = 1e-8) -> Tuple[bool, dict]:
    """Check orthogonality and det=+1; return diagnostics."""
    M = np.asarray(Rm, dtype=float).reshape(3, 3)
    I = np.eye(3)
    MtM = M.T @ M
    ortho = np.allclose(MtM, I, atol=tol)
    det = float(np.linalg.det(M))
    det1 = np.isclose(det, 1.0, atol=1e-8)
    rows = [float(np.linalg.norm(M[i])) for i in range(3)]
    cols = [float(np.linalg.norm(M[:, j])) for j in range(3)]
    return ortho and det1, {
        "det": det,
        "row_norms": rows,
        "col_norms": cols,
        "ortho_err": float(np.linalg.norm(MtM - I)),
    }

# ---------------------------------------------------------------------
# Sequences & primitive rotations
# ---------------------------------------------------------------------

# Tait–Bryan (all different) and proper Euler (first==last)
VALID_TB = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
VALID_PROPER = ["zxz", "xyx", "yzy", "zyz", "xzx", "yxy"]
VALID_ALL = VALID_TB + VALID_PROPER


def _axis_unit(ch: str) -> np.ndarray:
    ch = ch.lower()
    if ch == "x":
        return np.array([1.0, 0.0, 0.0])
    if ch == "y":
        return np.array([0.0, 1.0, 0.0])
    if ch == "z":
        return np.array([0.0, 0.0, 1.0])
    raise ValueError(f"Bad axis: {ch!r}")


def make_basic(axis: str, angle: float, degrees: bool = False) -> R:
    """Basic rotation about a global cartesian axis (Eq. 2.20–2.22 & 2.129–2.131)."""
    ang = math.radians(angle) if degrees else float(angle)
    return R.from_rotvec(_axis_unit(axis) * ang)

# ---------------------------------------------------------------------
# Composition / decomposition (global vs local)
# ---------------------------------------------------------------------

def compose_global(seq: str, angles: Sequence[float], degrees: bool = False) -> R:
    """
    Global/extrinsic composition: apply rotations about FIXED axes in the listed order.
    With SciPy's Rotation multiplication, left-to-right composition matches:
        R_total = R3 * R2 * R1
    so that a point p transforms as p' = R_total * p.
    """
    seq = seq.lower()
    if seq not in VALID_ALL:
        raise ValueError(f"sequence must be one of {VALID_ALL}")
    if len(angles) != 3:
        raise ValueError("angles must be length-3")

    Rlist = [make_basic(ax, ang, degrees=degrees) for ax, ang in zip(seq, angles)]
    Rtot = Rlist[2] * (Rlist[1] * Rlist[0])
    return Rtot


def compose_local(seq: str, angles: Sequence[float], degrees: bool = False) -> R:
    """
    Local/intrinsic composition: rotate about MOVING axes.
    Equivalent to reversed-order global rotations with reversed angles order:
        local(seq, a1,a2,a3) == global(seq[::-1], a3,a2,a1)
    """
    return compose_global(seq[::-1], angles[::-1], degrees=degrees)


def decompose(Robj: R, seq: str, convention: str = "global", degrees: bool = False) -> np.ndarray:
    """
    Extract angles for a given sequence and convention.
    For local (intrinsic) sequences we use the same reverse trick to be consistent
    with compose_local semantics.
    """
    seq = seq.lower()
    if convention not in {"global", "local"}:
        raise ValueError("convention must be 'global' or 'local'")
    if convention == "global":
        return Robj.as_euler(seq, degrees=degrees)

    # local: R = local(seq, a1,a2,a3)  ==  global(seq[::-1], a3,a2,a1)
    # to recover [a1,a2,a3], decompose with reversed sequence and reverse the result
    ang_rev = Robj.as_euler(seq[::-1], degrees=degrees)
    return ang_rev[::-1]

# ---------------------------------------------------------------------
# Repeat / power via logarithm map (Eq. 2.84–2.85 and related examples)
# ---------------------------------------------------------------------

def rotation_power(Robj: R, m: int) -> R:
    """Return R^m for integer m using axis-angle scaling (log/exp on SO(3))."""
    if not isinstance(m, (int, np.integer)):
        raise ValueError("m must be integer")
    rotvec = Robj.as_rotvec()
    ang = float(np.linalg.norm(rotvec))
    if ang < 1e-12:
        return R.identity()
    axis = rotvec / ang
    return R.from_rotvec(axis * (float(m) * ang))

# ---------------------------------------------------------------------
# Alignment (Eqs. 2.99–2.106): align body x-axis with a target vector
# ---------------------------------------------------------------------

def align_x_to_vector(u: Sequence[float]) -> R:
    """
    Build a rotation whose body x-axis aligns with u, with body y-axis chosen in the
    global (X,Y)-plane when possible; columns of the returned matrix are the body
    axes expressed in the global frame (right-handed).
    """
    u = np.asarray(u, dtype=float)
    n = np.linalg.norm(u)
    if n < 1e-12:
        raise ValueError("vector u must be nonzero")
    i = u / n  # body x-axis in global
    K = np.array([0.0, 0.0, 1.0])

    # pick j by projecting global K to plane normal i
    j = K - (K @ i) * i
    if np.linalg.norm(j) < 1e-12:  # u parallel to Z; fall back to Y
        j = np.array([0.0, 1.0, 0.0]) if i[2] >= 0.0 else np.array([0.0, -1.0, 0.0])
    j = j / np.linalg.norm(j)
    k = np.cross(i, j)

    # re-orthonormalize lightly to counter numeric drift
    j = np.cross(k, i); j = j / np.linalg.norm(j)
    k = np.cross(i, j)

    Rm = np.column_stack([i, j, k])  # columns are body axes in global
    return R.from_matrix(Rm)

# ---------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------

def save_matrix_csv(path: str, Robj: R) -> None:
    """Save a 3×3 matrix to CSV (creates parent directories)."""
    ensure_dir(os.path.dirname(path) or ".")
    np.savetxt(path, Robj.as_matrix(), delimiter=",", fmt="%.9f")


def save_series_csv(path: str, t: np.ndarray, data: np.ndarray, header: str = "t, a1, a2, a3") -> None:
    """
    Save a time series to CSV. Expects t shape (N,), data shape (N, K).
    """
    ensure_dir(os.path.dirname(path) or ".")
    arr = np.column_stack([t.reshape(-1), data])
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.9f")
