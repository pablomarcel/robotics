# acceleration/tools/quaternion.py
"""
Quaternion helpers for **acceleration kinematics**.

What this module provides
-------------------------
- Core quaternion ops (unit-normalized):
    * q_normalize(q)                 → (4,)
    * q_conj(q)                      → (4,)
    * q_mul(p, q)                    → (4,)  (Hamilton product, [w,x,y,z])
    * L(q), R(q)                     → (4,4) left/right-multiply matrices
    * R_from_q(q)                    → (3,3) rotation matrix
    * q_from_R(R)                    → (4,)  unit quaternion (w≥0)

- Angular velocity / acceleration mappings (world-frame convention):
    Given unit q = [w, x, y, z], define A(q) ∈ R^{4×3}:
        A(q) = [[-x,   -y,   -z],
                [ w,   -z,    y],
                [ z,    w,   -x],
                [-y,    x,    w]]
    Then for **unit** q:
        ω = 2 * A(q)^T q̇
        α = 2 * ( A(q)^T q̈ + (dA/dt)^T q̇ )
    where (dA/dt) is A with (ẇ, ẋ, ẏ, ż) substituted in place of (w, x, y, z).

    Functions:
      * omega_from_qdot(q, qdot)     → (3,)
      * alpha_from_qddot(q, qdot, qddot) → (3,)

- OO façade:
    * QuaternionKinematics(q, qdot, qddot)
        .omega()  → ω
        .alpha()  → α
        .update(...) → new instance (immutable style)

Conventions & Frames
--------------------
- Quaternions are **[w, x, y, z]** and represent the active rotation from a
  *body* frame to the *world* frame, i.e., R = R(q) maps body vectors into world.
- The ω and α returned here are expressed in the **world frame**. This matches
  the rate relation q̇ = 0.5 * S(ω) q with
      S(ω) = [[0, -ωᵀ],
              [ω, -[ω]^]]
  and leads to the linear map ω = 2 A(q)^T q̇ for unit q.
- If you need body-frame angular rates, simply rotate:
      ω_body = R(q)^T ω_world
      α_body = R(q)^T α_world    (plus ω× term if you’re transporting between frames)

Shapes
------
- q, qdot, qddot: (4,)
- ω, α: (3,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from ..utils import ensure_shape, asvec, skew


# -----------------------------------------------------------------------------
# Core quaternion operations
# -----------------------------------------------------------------------------

@ensure_shape(4,)
def q_normalize(q: Sequence[float] | np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return q/||q|| with a safe guard for tiny norms; ensures w≥0 for uniqueness."""
    q = asvec(q, 4).astype(float)
    n = float(np.linalg.norm(q))
    if n < eps:
        raise ValueError("Cannot normalize near-zero quaternion.")
    qn = q / n
    # Enforce w >= 0 canonical form (common for uniqueness in tests/docs)
    if qn[0] < 0.0:
        qn = -qn
    return qn


@ensure_shape(4,)
def q_conj(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Conjugate [w, x, y, z] -> [w, -x, -y, -z]."""
    q = asvec(q, 4)
    return np.array([q[0], -q[1], -q[2], -q[3]], float)


@ensure_shape(4,)
def q_mul(p: Sequence[float] | np.ndarray, q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Hamilton product p ⊗ q with [w, x, y, z] convention."""
    pw, px, py, pz = asvec(p, 4)
    qw, qx, qy, qz = asvec(q, 4)
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], float)


@ensure_shape(4, 4)
def L(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Left-multiply matrix: L(q) r = q ⊗ r."""
    w, x, y, z = asvec(q, 4)
    return np.array([
        [ w, -x, -y, -z],
        [ x,  w, -z,  y],
        [ y,  z,  w, -x],
        [ z, -y,  x,  w],
    ], float)


@ensure_shape(4, 4)
def R(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Right-multiply matrix: R(q) r = r ⊗ q."""
    w, x, y, z = asvec(q, 4)
    return np.array([
        [ w, -x, -y, -z],
        [ x,  w,  z, -y],
        [ y, -z,  w,  x],
        [ z,  y, -x,  w],
    ], float)


@ensure_shape(3, 3)
def R_from_q(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Rotation matrix from unit quaternion [w, x, y, z], body→world."""
    w, x, y, z = q_normalize(q)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
    ], float)


@ensure_shape(4,)
def q_from_R(Rm: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Unit quaternion from rotation matrix (w≥0 canonical)."""
    Rm = np.asarray(Rm, float).reshape(3, 3)
    tr = float(np.trace(Rm))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (Rm[2, 1] - Rm[1, 2]) / S
        y = (Rm[0, 2] - Rm[2, 0]) / S
        z = (Rm[1, 0] - Rm[0, 1]) / S
    else:
        i = int(np.argmax([Rm[0, 0], Rm[1, 1], Rm[2, 2]]))
        if i == 0:
            S = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2.0
            w = (Rm[2, 1] - Rm[1, 2]) / S
            x = 0.25 * S
            y = (Rm[0, 1] + Rm[1, 0]) / S
            z = (Rm[0, 2] + Rm[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 - Rm[0, 0] + Rm[1, 1] - Rm[2, 2]) * 2.0
            w = (Rm[0, 2] - Rm[2, 0]) / S
            x = (Rm[0, 1] + Rm[1, 0]) / S
            y = 0.25 * S
            z = (Rm[1, 2] + Rm[2, 1]) / S
        else:
            S = np.sqrt(1.0 - Rm[0, 0] - Rm[1, 1] + Rm[2, 2]) * 2.0
            w = (Rm[1, 0] - Rm[0, 1]) / S
            x = (Rm[0, 2] + Rm[2, 0]) / S
            y = (Rm[1, 2] + Rm[2, 1]) / S
            z = 0.25 * S
    return q_normalize([w, x, y, z])


# -----------------------------------------------------------------------------
# Linear ω/α maps via A(q)
# -----------------------------------------------------------------------------

@ensure_shape(4, 3)
def A_of_q(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Matrix A(q) such that (for **unit** q):  q̇ = 0.5 * A(q) ω   and   ω = 2 * A(q)^T q̇

    q = [w, x, y, z] (unit). Angular velocity is **world-frame**.
    """
    w, x, y, z = q_normalize(q)
    return np.array([
        [-x,  -y,  -z],
        [ w,  -z,   y],
        [ z,   w,  -x],
        [-y,   x,   w],
    ], float)


@ensure_shape(4, 3)
def Adot_of_q_qdot(q: Sequence[float] | np.ndarray,
                   qdot: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Time derivative Ȧ(q, q̇). Constructed by replacing (w, x, y, z) with (ẇ, ẋ, ẏ, ż)
    in the linear terms of A(q). Assumes q is kept unit (numerically normalize).
    """
    w, x, y, z = q_normalize(q)
    wd, xd, yd, zd = asvec(qdot, 4)
    return np.array([
        [-xd,  -yd,  -zd],
        [ wd,  -zd,   yd],
        [ zd,   wd,  -xd],
        [-yd,   xd,   wd],
    ], float)


@ensure_shape(3,)
def omega_from_qdot(q: Sequence[float] | np.ndarray,
                    qdot: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Angular velocity (world-frame) from quaternion and quaternion rate:
        ω = 2 * A(q)^T q̇
    """
    q = q_normalize(q)
    qd = asvec(qdot, 4)
    return 2.0 * (A_of_q(q).T @ qd)


@ensure_shape(3,)
def alpha_from_qddot(q: Sequence[float] | np.ndarray,
                     qdot: Sequence[float] | np.ndarray,
                     qddot: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Angular acceleration (world-frame) from q, q̇, q̈:
        α = 2 * ( A(q)^T q̈ + Ȧ(q,q̇)^T q̇ )
    """
    q = q_normalize(q)
    qd = asvec(qdot, 4)
    qdd = asvec(qddot, 4)
    A = A_of_q(q)
    Ad = Adot_of_q_qdot(q, qd)
    return 2.0 * (A.T @ qdd + Ad.T @ qd)


# -----------------------------------------------------------------------------
# OO façade
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class QuaternionKinematics:
    """
    Immutable quaternion kinematics helper (world-frame angular quantities).

    Attributes
    ----------
    q     : (4,) unit quaternion [w, x, y, z] (body→world)
    qdot  : (4,) quaternion rate
    qddot : (4,) quaternion acceleration

    Methods
    -------
    omega()  → (3,) world angular velocity
    alpha()  → (3,) world angular acceleration
    update(q=None, qdot=None, qddot=None) → QuaternionKinematics
    """
    q: np.ndarray
    qdot: np.ndarray
    qddot: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", q_normalize(self.q))
        object.__setattr__(self, "qdot", asvec(self.qdot, 4))
        object.__setattr__(self, "qddot", asvec(self.qddot, 4))

    @classmethod
    def from_lists(cls,
                   q: Sequence[float],
                   qdot: Sequence[float],
                   qddot: Sequence[float]) -> "QuaternionKinematics":
        return cls(np.asarray(q, float), np.asarray(qdot, float), np.asarray(qddot, float))

    @ensure_shape(3,)
    def omega(self) -> np.ndarray:
        return omega_from_qdot(self.q, self.qdot)

    @ensure_shape(3,)
    def alpha(self) -> np.ndarray:
        return alpha_from_qddot(self.q, self.qdot, self.qddot)

    def update(self,
               q: Sequence[float] | np.ndarray | None = None,
               qdot: Sequence[float] | np.ndarray | None = None,
               qddot: Sequence[float] | np.ndarray | None = None) -> "QuaternionKinematics":
        """Return a new instance with provided fields replaced (keeps unit-q)."""
        return QuaternionKinematics(
            q_normalize(self.q if q is None else q),
            asvec(self.qdot if qdot is None else qdot, 4),
            asvec(self.qddot if qddot is None else qddot, 4),
        )


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    # core ops
    "q_normalize",
    "q_conj",
    "q_mul",
    "L",
    "R",
    "R_from_q",
    "q_from_R",
    # linear maps
    "A_of_q",
    "Adot_of_q_qdot",
    "omega_from_qdot",
    "alpha_from_qddot",
    # OO façade
    "QuaternionKinematics",
]
