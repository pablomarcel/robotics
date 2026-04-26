"""
orientation.core
----------------
Object-oriented core math for orientation kinematics (SO(3)).

This module defines four primary classes:

- AxisAngle       : φ about unit axis û   (Rodrigues / Euler axis–angle)
- RodriguesVector : w = û * tan(φ/2)
- Quaternion      : unit quaternion (Euler parameters) [e0, e1, e2, e3] (scalar-first)
- SO3             : rotation_kinematics group element with a 3x3 matrix representation

All classes are immutable-ish (dataclasses with tuples/np arrays) and provide
clear conversion methods so they are easy to unit-test.

Numerical notes
---------------
- Conversions handle near-zero and near-π angles with safe branches.
- All inputs are not assumed normalized; constructors normalize internally.
- Tolerances controlled by EPS and use of np.clip to keep values in domain.

References (matching book equations)
------------------------------------
Axis-angle ↔ matrix   : Eqs. 3.3–3.7, 3.10–3.16
Matrix → axis-angle   : Eqs. 3.10–3.11, 3.17–3.18
Euler parameters (quat) ↔ matrix : Eq. 3.140 and companions
Rodrigues vector      : Eqs. 3.201–3.206
Exponential map       : Eq. 3.187 (handled in utils.expm_so3)

Author: Robotics project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants & helpers (kept private; tests may probe via methods)
# ---------------------------------------------------------------------------

EPS = 1e-12
PI = np.pi


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n < EPS:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n


def _skew(u: np.ndarray) -> np.ndarray:
    ux, uy, uz = u
    return np.array([[0.0, -uz, uy],
                     [uz, 0.0, -ux],
                     [-uy, ux, 0.0]], dtype=float)


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """Project near-rotation_kinematics matrix to SO(3) via SVD: closest in Frobenius norm."""
    U, _, Vt = np.linalg.svd(R)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:  # enforce det +1
        U[:, -1] *= -1
        Rproj = U @ Vt
    return Rproj


# ---------------------------------------------------------------------------
# Axis–Angle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisAngle:
    """Axis–angle (Rodrigues) representation.

    Parameters
    ----------
    phi : float
        Rotation angle (radians). Can be outside [-π, π]; stored unchanged.
    u : np.ndarray
        3-vector; need not be unit. Will be normalized internally.

    Methods
    -------
    as_matrix() -> np.ndarray
    to_quaternion() -> Quaternion
    to_rodrigues() -> RodriguesVector
    compose(other: AxisAngle) -> SO3
    static from_matrix(R: np.ndarray) -> AxisAngle
    """

    phi: float
    u: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "u", _unit(np.asarray(self.u, dtype=float)))

    # Eq. (3.5) / (3.16): R = I cos φ + û û^T vers φ + û~ sin φ
    def as_matrix(self) -> np.ndarray:
        phi = float(self.phi)
        u = self.u
        ux = _skew(u)
        uuT = np.outer(u, u)
        c = np.cos(phi)
        s = np.sin(phi)
        vers = 1.0 - c  # vers(φ)
        R = c * np.eye(3) + vers * uuT + s * ux
        return R

    # Robust matrix → axis-angle using trace relation and stable branches.
    @staticmethod
    def from_matrix(R: np.ndarray) -> "AxisAngle":
        R = np.asarray(R, dtype=float).reshape(3, 3)
        # Project to SO(3) for numerical stability
        R = _project_to_so3(R)

        tr = float(np.trace(R))
        # angle from trace: tr = 1 + 2 cos φ
        cos_phi = np.clip(0.5 * (tr - 1.0), -1.0, 1.0)
        phi = float(np.arccos(cos_phi))

        # ----- very small angle: identity (axis undefined; pick a conventional axis) -----
        if phi < 1e-8:
            # Try to derive a direction from the skew part if there is tiny numeric signal;
            # otherwise pick a conventional axis (x-axis). Do NOT normalize a near-zero vector.
            w = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]) * 0.5
            n = float(np.linalg.norm(w))
            if n > 1e-12:
                u = w / n
            else:
                u = np.array([1.0, 0.0, 0.0])  # arbitrary (any unit axis is valid at φ=0)
            return AxisAngle(phi=0.0, u=u)

        # ----- near π: use a stable branch that chooses the largest diagonal component -----
        if abs(PI - phi) < 1e-6:
            # Build axis from the diagonal with the largest value to avoid division by tiny numbers.
            # This mirrors a standard stable branch used in quat-from-matrix routines.
            diag = np.diag(R)
            i = int(np.argmax(diag))  # 0 -> x, 1 -> y, 2 -> z
            if i == 0:
                x = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 0.5
                if x < 1e-12:
                    # fallback to avoid division by ~0
                    u = np.array([1.0, 0.0, 0.0])
                else:
                    y = (R[0, 1] + R[1, 0]) / (4.0 * x)
                    z = (R[0, 2] + R[2, 0]) / (4.0 * x)
                    u = np.array([x, y, z], dtype=float)
            elif i == 1:
                y = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 0.5
                if y < 1e-12:
                    u = np.array([0.0, 1.0, 0.0])
                else:
                    x = (R[0, 1] + R[1, 0]) / (4.0 * y)
                    z = (R[1, 2] + R[2, 1]) / (4.0 * y)
                    u = np.array([x, y, z], dtype=float)
            else:
                z = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 0.5
                if z < 1e-12:
                    u = np.array([0.0, 0.0, 1.0])
                else:
                    x = (R[0, 2] + R[2, 0]) / (4.0 * z)
                    y = (R[1, 2] + R[2, 1]) / (4.0 * z)
                    u = np.array([x, y, z], dtype=float)

            # Normalize the axis and return
            return AxisAngle(phi=phi, u=_unit(u))

        # ----- general case: use û~ = (R - R^T) / (2 sin φ) -----
        u_tilde = (R - R.T) / (2.0 * np.sin(phi))
        u = np.array([u_tilde[2, 1], u_tilde[0, 2], u_tilde[1, 0]])
        return AxisAngle(phi=phi, u=_unit(u))

    def to_quaternion(self) -> "Quaternion":
        return Quaternion.from_axis_angle(self.phi, self.u)

    def to_rodrigues(self) -> "RodriguesVector":
        # w = û tan(φ/2)
        return RodriguesVector(self.u * np.tan(self.phi / 2.0))

    def compose(self, other: "AxisAngle") -> "SO3":
        """Return SO3 representing R_other * R_self."""
        return SO3(self.as_matrix()).compose(SO3(other.as_matrix()))


# ---------------------------------------------------------------------------
# Rodrigues vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RodriguesVector:
    """Rodrigues vector w = û tan(φ/2). Compact axis–angle container.

    Methods
    -------
    as_matrix() -> np.ndarray
    to_axis_angle() -> AxisAngle
    to_quaternion() -> Quaternion
    """

    w: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "w", np.asarray(self.w, dtype=float).reshape(3))

    def to_axis_angle(self) -> AxisAngle:
        w = self.w
        w2 = float(np.dot(w, w))
        denom = 1.0 + w2
        # cos(φ/2) = 1/sqrt(1+w²), sin(φ/2)=||w||/sqrt(1+w²)
        e0 = 1.0 / np.sqrt(denom)
        s = _norm(w) / np.sqrt(denom)
        phi = 2.0 * np.arctan2(s, e0)  # stable
        u = w / (np.tan(phi / 2.0) + EPS) if phi > 1e-12 else np.array([1.0, 0.0, 0.0])
        return AxisAngle(phi=phi, u=_unit(u))

    def to_quaternion(self) -> "Quaternion":
        aa = self.to_axis_angle()
        return aa.to_quaternion()

    # Eq. (3.205): matrix from Rodrigues vector (rearranged to classic form)
    def as_matrix(self) -> np.ndarray:
        w = self.w
        w2 = float(np.dot(w, w))
        I = np.eye(3)
        wx = _skew(w)
        # Correct closed form:
        # R = ((1 - w^T w) I + 2 [w]_x + 2 w w^T) / (1 + w^T w)
        return ((1.0 - w2) * I + 2.0 * wx + 2.0 * np.outer(w, w)) / (1.0 + w2)


# ---------------------------------------------------------------------------
# Quaternion (Euler parameters) — scalar-first
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quaternion:
    """Unit quaternion (Euler parameters) with scalar-first storage.

    Fields
    ------
    e0 : float
    e1, e2, e3 : float (vector part)

    Methods
    -------
    normalized(), conj(), as_matrix(), rotate(v), multiply(q)
    to_axis_angle(), to_rodrigues()
    class from_matrix(R), from_axis_angle(phi, u)
    """

    e0: float
    e1: float
    e2: float
    e3: float

    # --- construction helpers ---

    def as_np(self) -> np.ndarray:
        return np.array([self.e0, self.e1, self.e2, self.e3], dtype=float)

    def norm(self) -> float:
        return float(np.linalg.norm(self.as_np()))

    def normalized(self) -> "Quaternion":
        v = self.as_np()
        n = np.linalg.norm(v)
        if n < EPS:
            raise ValueError("Zero quaternion cannot be normalized.")
        v = v / n
        return Quaternion(*map(float, v))

    def conj(self) -> "Quaternion":
        return Quaternion(self.e0, -self.e1, -self.e2, -self.e3)

    # Hamilton product (this ⊗ q)
    def multiply(self, q: "Quaternion") -> "Quaternion":
        a0, a1, a2, a3 = self.e0, self.e1, self.e2, self.e3
        b0, b1, b2, b3 = q.e0, q.e1, q.e2, q.e3
        return Quaternion(
            a0*b0 - a1*b1 - a2*b2 - a3*b3,
            a0*b1 + a1*b0 + a2*b3 - a3*b2,
            a0*b2 - a1*b3 + a2*b0 + a3*b1,
            a0*b3 + a1*b2 - a2*b1 + a3*b0,
        )

    # Eq. (3.140) and (3.256): quaternion to matrix
    def as_matrix(self) -> np.ndarray:
        q = self.normalized()
        e0, e1, e2, e3 = q.e0, q.e1, q.e2, q.e3
        e = np.array([e1, e2, e3])
        eeT = np.outer(e, e)
        ex = _skew(e)
        R = (e0*e0 - np.dot(e, e)) * np.eye(3) + 2 * eeT + 2 * e0 * ex
        # Project to SO(3) to counteract tiny drift from float ops
        return _project_to_so3(R)

    def rotate(self, v: np.ndarray) -> np.ndarray:
        """Rotate a 3-vector by this quaternion."""
        R = self.as_matrix()
        return R @ np.asarray(v, dtype=float)

    def to_axis_angle(self) -> AxisAngle:
        q = self.normalized()
        e0 = np.clip(q.e0, -1.0, 1.0)
        phi = 2.0 * float(np.arccos(e0))
        s = np.sqrt(max(1.0 - e0 * e0, 0.0))
        if s < 1e-12:
            u = np.array([1.0, 0.0, 0.0])  # arbitrary when angle ~ 0
        else:
            u = np.array([q.e1, q.e2, q.e3]) / s
        return AxisAngle(phi=phi, u=_unit(u))

    def to_rodrigues(self) -> RodriguesVector:
        q = self.normalized()
        # Ensure canonical quaternion (non-negative scalar part) so Rodrigues is unique
        if q.e0 < 0:
            q = Quaternion(-q.e0, -q.e1, -q.e2, -q.e3)
        if abs(q.e0) < EPS:
            return RodriguesVector(np.array([np.inf, np.inf, np.inf]))
        return RodriguesVector(np.array([q.e1, q.e2, q.e3]) / q.e0)

    # --- alt constructors ---

    @staticmethod
    def from_axis_angle(phi: float, u: np.ndarray) -> "Quaternion":
        u = _unit(np.asarray(u, dtype=float))
        half = 0.5 * float(phi)
        e0 = float(np.cos(half))
        s = float(np.sin(half))
        e = u * s
        return Quaternion(e0, e[0], e[1], e[2]).normalized()

    @staticmethod
    def from_matrix(R: np.ndarray) -> "Quaternion":
        """Robust matrix → quaternion (scalar-first)."""
        R = _project_to_so3(np.asarray(R, dtype=float).reshape(3, 3))
        tr = float(np.trace(R))
        if tr > 0:
            t = np.sqrt(max(tr + 1.0, 0.0)) * 2.0  # 4*e0
            e0 = 0.25 * t
            e1 = (R[2, 1] - R[1, 2]) / t
            e2 = (R[0, 2] - R[2, 0]) / t
            e3 = (R[1, 0] - R[0, 1]) / t
        else:
            # find dominant diagonal
            i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
            if i == 0:
                t = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
                e0 = (R[2, 1] - R[1, 2]) / t
                e1 = 0.25 * t
                e2 = (R[0, 1] + R[1, 0]) / t
                e3 = (R[0, 2] + R[2, 0]) / t
            elif i == 1:
                t = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
                e0 = (R[0, 2] - R[2, 0]) / t
                e1 = (R[0, 1] + R[1, 0]) / t
                e2 = 0.25 * t
                e3 = (R[1, 2] + R[2, 1]) / t
            else:
                t = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
                e0 = (R[1, 0] - R[0, 1]) / t
                e1 = (R[0, 2] + R[2, 0]) / t
                e2 = (R[1, 2] + R[2, 1]) / t
                e3 = 0.25 * t
        return Quaternion(e0, e1, e2, e3).normalized()


# ---------------------------------------------------------------------------
# SO(3) wrapper (matrix representation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SO3:
    """SO(3) rotation_kinematics class with a 3×3 matrix representation.

    Construct directly from a 3×3 (validated and projected), or use the
    convenience constructors.

    Methods
    -------
    compose(other) -> SO3
    inverse() -> SO3
    act(v) -> np.ndarray
    angle() -> float
    axis() -> np.ndarray
    to_axis_angle() / to_quaternion() / to_rodrigues()
    class from_axis_angle(phi, u)
    class from_quaternion(q)
    class from_rodrigues(w)
    identity()
    """

    R: np.ndarray

    def __post_init__(self):
        R = _project_to_so3(np.asarray(self.R, dtype=float).reshape(3, 3))
        object.__setattr__(self, "R", R)

    # --- composition & group ops ---

    def compose(self, other: "SO3") -> "SO3":
        """Return self * other (apply 'other' then 'self')."""
        return SO3(self.R @ other.R)

    def inverse(self) -> "SO3":
        return SO3(self.R.T)

    def act(self, v: np.ndarray) -> np.ndarray:
        return self.R @ np.asarray(v, dtype=float)

    # --- geometric accessors ---

    def to_axis_angle(self) -> AxisAngle:
        return AxisAngle.from_matrix(self.R)

    def to_quaternion(self) -> Quaternion:
        return Quaternion.from_matrix(self.R)

    def to_rodrigues(self) -> RodriguesVector:
        return self.to_quaternion().to_rodrigues()

    def angle(self) -> float:
        aa = self.to_axis_angle()
        return float(abs(aa.phi))

    def axis(self) -> np.ndarray:
        return self.to_axis_angle().u

    # --- constructors ---

    @staticmethod
    def identity() -> "SO3":
        return SO3(np.eye(3))

    @staticmethod
    def from_axis_angle(phi: float, u: np.ndarray) -> "SO3":
        return SO3(AxisAngle(phi, u).as_matrix())

    @staticmethod
    def from_quaternion(q: Quaternion) -> "SO3":
        return SO3(q.as_matrix())

    @staticmethod
    def from_rodrigues(w: np.ndarray | RodriguesVector) -> "SO3":
        rv = w if isinstance(w, RodriguesVector) else RodriguesVector(np.asarray(w, dtype=float))
        return SO3(rv.as_matrix())
