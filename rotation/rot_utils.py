# =============================
# File: rotation/rot_utils.py
# =============================
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy.spatial.transform import Rotation as _R
except Exception as _e:  # pragma: no cover
    raise RuntimeError("SciPy is required: pip install scipy") from _e

R = _R  # alias

# ---------- basic helpers ----------

def ensure_dir(p: str) -> None:
    if p == "":
        return
    os.makedirs(p, exist_ok=True)

def parse_floats(csv: str | Sequence[float]) -> List[float]:
    if isinstance(csv, (list, tuple, np.ndarray)):
        return [float(x) for x in csv]
    parts = [s for s in str(csv).replace(" ", "").split(",") if s != ""]
    return [float(x) for x in parts]

def hat(v: ArrayLike) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=float)

def vee(M: ArrayLike) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return np.array([M[2,1] - M[1,2],
                     M[0,2] - M[2,0],
                     M[1,0] - M[0,1]], dtype=float) * 0.5

def is_rotmat(Rm: ArrayLike, tol: float = 1e-8) -> Tuple[bool, dict]:
    M = np.asarray(Rm, dtype=float).reshape(3,3)
    I = np.eye(3)
    MtM = M.T @ M
    ortho = np.allclose(MtM, I, atol=tol)
    det = float(np.linalg.det(M))
    det1 = np.isclose(det, 1.0, atol=1e-8)
    rows = [float(np.linalg.norm(M[i])) for i in range(3)]
    cols = [float(np.linalg.norm(M[:,j])) for j in range(3)]
    return ortho and det1, {
        "det": det,
        "row_norms": rows,
        "col_norms": cols,
        "ortho_err": float(np.linalg.norm(MtM - I)),
    }

# ---------- sequences ----------

VALID_TB = ["xyz","xzy","yxz","yzx","zxy","zyx"]
VALID_PROPER = ["zxz","xyx","yzy","zyz","xzx","yxy"]
VALID_ALL = VALID_TB + VALID_PROPER

# ---------- composition / decomposition ----------

def compose_global(seq: str, angles: Sequence[float], degrees: bool = False) -> R:
    """Extrinsic (fixed axes)."""
    s = seq.lower()
    if s not in VALID_ALL:
        raise ValueError(f"sequence must be one of {VALID_ALL}")
    if len(angles) != 3:
        raise ValueError("angles must be length-3")
    return R.from_euler(s, angles, degrees=degrees)

def compose_local(seq: str, angles: Sequence[float], degrees: bool = False) -> R:
    """Intrinsic (moving axes) — SciPy uppercase."""
    S = seq.upper()
    if seq.lower() not in VALID_ALL:
        raise ValueError(f"sequence must be one of {VALID_ALL}")
    if len(angles) != 3:
        raise ValueError("angles must be length-3")
    return R.from_euler(S, angles, degrees=degrees)

def decompose(Robj: R, seq: str, convention: str = "global", degrees: bool = False) -> np.ndarray:
    """Mirror compose_* for extraction."""
    if convention not in {"global","local"}:
        raise ValueError("convention must be 'global' or 'local'")
    if convention == "global":
        return Robj.as_euler(seq.lower(), degrees=degrees)
    else:
        return Robj.as_euler(seq.upper(), degrees=degrees)

# ---------- ω ↔ q̇ mapping (central differences, radians internally) ----------

def _omega_from_Rdot(Robj: R, Rdot: np.ndarray, frame: str = 'body') -> np.ndarray:
    Rm = Robj.as_matrix()
    if frame == 'space':
        Om = Rdot @ Rm.T
    elif frame == 'body':
        Om = Rm.T @ Rdot
    else:
        raise ValueError("frame must be 'space' or 'body'")
    Om = 0.5*(Om - Om.T)
    return vee(Om)

def build_rate_map(seq: str, angles: Sequence[float],
                   convention: str = 'global', degrees: bool = False,
                   frame: str = 'body') -> np.ndarray:
    """Return M(q) so that ω = M(q) q̇."""
    ang = np.asarray(angles, float)
    q_rad = np.deg2rad(ang) if degrees else ang
    seq_tag = seq.lower() if convention == 'global' else seq.upper()

    R0 = R.from_euler(seq_tag, q_rad, degrees=False)
    h = 1e-8
    M = np.zeros((3,3), float)
    for j in range(3):
        qp = q_rad.copy(); qp[j] += h
        qm = q_rad.copy(); qm[j] -= h
        Rp = R.from_euler(seq_tag, qp, degrees=False)
        Rm = R.from_euler(seq_tag, qm, degrees=False)
        Rdot = (Rp.as_matrix() - Rm.as_matrix())/(2*h)
        M[:, j] = _omega_from_Rdot(R0, Rdot, frame=frame)
    return M

def omega_from_rates(seq: str, angles: Sequence[float], rates: Sequence[float],
                     convention: str = 'global', degrees: bool = False,
                     frame: str = 'body') -> np.ndarray:
    M = build_rate_map(seq, angles, convention=convention, degrees=degrees, frame=frame)
    qdot = np.asarray(rates, float)
    if degrees:
        qdot = np.deg2rad(qdot)
    w = M @ qdot
    return np.rad2deg(w) if degrees else w

def rates_from_omega(seq: str, angles: Sequence[float], omega: Sequence[float],
                     convention: str = 'global', degrees: bool = False,
                     frame: str = 'body') -> np.ndarray:
    M = build_rate_map(seq, angles, convention=convention, degrees=degrees, frame=frame)
    w = np.asarray(omega, float)
    if degrees:
        w = np.deg2rad(w)
    qdot = np.linalg.pinv(M) @ w
    return np.rad2deg(qdot) if degrees else qdot

# ---------- repeat & alignment ----------

def rotation_power(Robj: R, m: int) -> R:
    if not isinstance(m, (int, np.integer)):
        raise ValueError("m must be integer")
    rotvec = Robj.as_rotvec()
    ang = float(np.linalg.norm(rotvec))
    if ang < 1e-12:
        return R.identity()
    axis = rotvec/ang
    return R.from_rotvec(axis * (float(m) * ang))

def align_x_to_vector(u: Sequence[float]) -> R:
    u = np.asarray(u, float)
    n = np.linalg.norm(u)
    if n < 1e-12:
        raise ValueError("vector u must be nonzero")
    i = u/n
    K = np.array([0.0,0.0,1.0])
    j = K - (K @ i) * i
    if np.linalg.norm(j) < 1e-12:
        j = np.array([0.0,1.0,0.0]) if i[2] >= 0 else np.array([0.0,-1.0,0.0])
    j = j/np.linalg.norm(j)
    k = np.cross(i, j)
    j = np.cross(k, i); j = j/np.linalg.norm(j)
    k = np.cross(i, j)
    Rm = np.column_stack([i, j, k])
    return R.from_matrix(Rm)

# ---------- CSV I/O ----------

def save_matrix_csv(path: str, Robj: R) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.savetxt(path, Robj.as_matrix(), delimiter=",", fmt="%.9f")

def save_series_csv(path: str, t: np.ndarray, data: np.ndarray, header: str = "t, a1, a2, a3") -> None:
    ensure_dir(os.path.dirname(path) or ".")
    arr = np.column_stack([t.reshape(-1), data])
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.9f")
