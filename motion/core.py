# motion/core.py
from __future__ import annotations

"""
Core math primitives for rigid-body motion kinematics (Chapter 4 scope).

Classes
-------
Rotation     : SO(3) rotation with axis–angle factories (Rodrigues).
SE3          : Homogeneous transform (R, t) with composition/inverse/apply.
Screw        : Screw motion parameterization {u, s, h, phi} → SE3.
PluckerLine  : Plücker line [u; rho], angle & distance, 6×6 action via SE3.
Plane        : Plane n·x = s (point distance).
DHLink       : Standard DH link (a, alpha, d, theta) → SE3.
KinematicChain : Product of DH links (forward kinematics).

Helpers
-------
hat(v), vee(S), normalize(v)

Key equation coverage (non-exhaustive)
--------------------------------------
Rotations/SE(3): 4.1–4.66, 4.75–4.90
Rodrigues:      4.221 (and classic 3.187, used in §4)
Screw motion:   4.206, 4.220–4.224 (û, s, h, φ → (R, t))
Plücker:        4.344–4.346 (build), 4.382 (SE3 action),
                4.388–4.390 (reciprocal product, angle, distance)
Planes:         4.391–4.397
DH FK:          standard A_i form widely used in §4
"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import math
import numpy as np


# ------------------------------- helpers -------------------------------------

def hat(v: np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix [v]_x that implements cross products: [v]_x w = v × w.
    """
    v = np.asarray(v, dtype=float).reshape(3)
    x, y, z = v
    return np.array([[0.0, -z,  y],
                     [z,  0.0, -x],
                     [-y, x,  0.0]], dtype=float)


def vee(S: np.ndarray) -> np.ndarray:
    """
    Inverse of hat for so(3) → R^3.
    """
    S = np.asarray(S, dtype=float).reshape(3, 3)
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Safe vector normalization. Returns a copy if norm==0 (no raise).
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        return v.copy()
    return v / n


# -------------------------------- Rotation -----------------------------------

@dataclass(frozen=True)
class Rotation:
    """
    SO(3) rotation.

    Notes
    -----
    - Axis–angle → matrix uses Rodrigues:
        R = I + sinφ [u]_x + (1 - cosφ) [u]_x^2 (cf. 4.221 / classic 3.187)
    """
    R: np.ndarray  # shape (3, 3)

    # ---- factories ----
    @staticmethod
    def from_axis_angle(axis: Iterable[float], phi: float) -> "Rotation":
        u = normalize(np.asarray(axis, float).reshape(3))
        K = hat(u)
        s, c = math.sin(phi), math.cos(phi)
        R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
        return Rotation(R)

    @staticmethod
    def Rx(theta: float) -> "Rotation":
        c, s = math.cos(theta), math.sin(theta)
        return Rotation(np.array([[1, 0, 0],
                                  [0, c, -s],
                                  [0, s,  c]], float))

    @staticmethod
    def Ry(theta: float) -> "Rotation":
        c, s = math.cos(theta), math.sin(theta)
        return Rotation(np.array([[ c, 0, s],
                                  [ 0, 1, 0],
                                  [-s, 0, c]], float))

    @staticmethod
    def Rz(theta: float) -> "Rotation":
        c, s = math.cos(theta), math.sin(theta)
        return Rotation(np.array([[c, -s, 0],
                                  [s,  c, 0],
                                  [0,  0, 1]], float))

    # ---- ops ----
    def inv(self) -> "Rotation":
        return Rotation(self.R.T)

    def as_matrix(self) -> np.ndarray:
        return self.R.copy()


# ---------------------------------- SE3 --------------------------------------

@dataclass(frozen=True)
class SE3:
    """
    Homogeneous transform T ∈ SE(3), represented as (R, t).

    Composition (G_T_C = G_T_B @ B_T_C) follows 4.59–4.66.
    Inverse: T^{-1} = [ R^T  -R^T t ; 0 0 0 1 ] (cf. 4.447).
    """
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    # ---- factories ----
    @staticmethod
    def identity() -> "SE3":
        return SE3(np.eye(3), np.zeros(3))

    @staticmethod
    def from_rt(R: Rotation, t: Iterable[float]) -> "SE3":
        return SE3(R.as_matrix(), np.asarray(t, float).reshape(3))

    @staticmethod
    def from_matrix(T: np.ndarray) -> "SE3":
        T = np.asarray(T, float).reshape(4, 4)
        return SE3(T[:3, :3].copy(), T[:3, 3].copy())

    # ---- conversions ----
    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    # ---- ops ----
    def inv(self) -> "SE3":
        RT = self.R.T
        return SE3(RT, -RT @ self.t)

    def __matmul__(self, other: "SE3") -> "SE3":
        """Composition (overriding '@')."""
        R = self.R @ other.R
        t = self.R @ other.t + self.t
        return SE3(R, t)

    # ---- application ----
    def apply(self, p: Iterable[float]) -> np.ndarray:
        """Apply to a 3D point."""
        p = np.asarray(p, float).reshape(3)
        return self.R @ p + self.t

    def apply_points(self, P: np.ndarray) -> np.ndarray:
        """Apply to an array of Nx3 points."""
        P = np.asarray(P, float)
        return (self.R @ P.T).T + self.t


# --------------------------------- Screw -------------------------------------

@dataclass(frozen=True)
class Screw:
    """
    Screw motion parameterization {u, s, h, phi} with transform builder.

    Parameters
    ----------
    u   : axis (not necessarily unit; normalized internally)
    s   : location vector (any point on the screw axis expressed in {G})
    h   : pitch (translation per radian)
    phi : rotation angle (radians)

    Formulas (cf. 4.206, 4.220–4.224)
    ---------------------------------
    R  = I + sinφ [û]_x + (1 - cosφ) [û]_x^2
    t  = (I - R) s + h φ û             (equivalently: s - R s + h φ û)
    T  = [[R, t], [0, 0, 0, 1]]
    """
    u: np.ndarray
    s: np.ndarray
    h: float
    phi: float

    def transform(self) -> SE3:
        u_hat = normalize(np.asarray(self.u, float).reshape(3))
        s_vec = np.asarray(self.s, float).reshape(3)
        R = Rotation.from_axis_angle(u_hat, self.phi).as_matrix()
        t = (np.eye(3) - R) @ s_vec + (self.h * self.phi) * u_hat
        return SE3(R, t)

    def to_matrix(self) -> np.ndarray:
        return self.transform().as_matrix()


# ------------------------------- Plücker line --------------------------------

@dataclass(frozen=True)
class PluckerLine:
    """
    Plücker line coordinates l = [u; rho], with ||u|| = 1 and u·rho = 0.

    Build from two points (4.344–4.346), transform by SE3 (4.382),
    and compute angle (4.389) / distance (4.390) between two lines.
    """
    u: np.ndarray    # (3,)
    rho: np.ndarray  # (3,)

    # ---- factories ----
    @staticmethod
    def from_points(p1: Iterable[float], p2: Iterable[float]) -> "PluckerLine":
        r1 = np.asarray(p1, float).reshape(3)
        r2 = np.asarray(p2, float).reshape(3)
        u = normalize(r2 - r1)
        rho = np.cross(r1, u)
        return PluckerLine(u, rho)

    # ---- relations ----
    def reciprocal_product(self, other: "PluckerLine") -> float:
        """
        Reciprocal product l2 × l1 = u2·rho1 + u1·rho2 (Eq. 4.388).
        """
        return float(self.u @ other.rho + other.u @ self.rho)

    def angle(self, other: "PluckerLine") -> float:
        """
        Robust principal angle α ∈ [0, π] between directions using:
            α = atan2( ||u1 × u2|| , u1 · u2 )   (more stable than asin near 90°)
        (cf. 4.389 for sinα; here we use atan2(sinα, cosα) for numerical stability.)
        """
        u1 = self.u / np.linalg.norm(self.u)
        u2 = other.u / np.linalg.norm(other.u)
        s = float(np.linalg.norm(np.cross(u1, u2)))
        c = float(np.clip(u1 @ u2, -1.0, 1.0))
        s = float(np.clip(s, 0.0, 1.0))
        return math.atan2(s, c)

    def distance(self, other: "PluckerLine") -> float:
        """
        Shortest distance d = |l2 × l1| / sinα  (Eq. 4.390).
        Returns NaN for parallel lines (sinα ≈ 0).
        """
        # Normalize directions for consistent numerics
        u1 = self.u / np.linalg.norm(self.u)
        u2 = other.u / np.linalg.norm(other.u)
        num = abs(self.reciprocal_product(other))
        den = float(np.linalg.norm(np.cross(u1, u2)))
        if den <= 1e-12:
            return float("nan")
        return float(num / den)

    # ---- actions ----
    def transform(self, T: SE3) -> "PluckerLine":
        """
        6×6 Plücker action induced by SE3 (Eq. 4.382).

        l' = Ad_T · l, where Ad_T = [[R, 0],[ [t]_x R, R ]].
        Implemented explicitly without materializing the full 6×6.
        """
        R, t = T.R, T.t
        u2 = R @ self.u
        # normalize to eliminate tiny drift (important for angle invariance tests)
        n = np.linalg.norm(u2)
        if n > 0:
            u2 = u2 / n
        rho2 = hat(t) @ u2 + R @ self.rho
        return PluckerLine(u2, rho2)


# ---------------------------------- Plane ------------------------------------

@dataclass(frozen=True)
class Plane:
    """
    Plane given by unit normal n and offset s:  n·x = s  (cf. 4.391).

    Distance to a point p: signed = n·p - s ; unsigned = |signed| (4.397).
    """
    n: np.ndarray
    s: float = 0.0

    @staticmethod
    def from_point_normal(point: Iterable[float], normal: Iterable[float]) -> "Plane":
        n_hat = normalize(np.asarray(normal, float).reshape(3))
        s = float(n_hat @ np.asarray(point, float).reshape(3))
        return Plane(n_hat, s)

    def distance_to_point(self, p: Iterable[float], *, signed: bool = True) -> float:
        p = np.asarray(p, float).reshape(3)
        d = float(self.n @ p - self.s)
        return d if signed else abs(d)


# -------------------------- DH Link & Kinematic Chain ------------------------

@dataclass(frozen=True)
class DHLink:
    """
    Standard Denavit–Hartenberg link: (a, alpha, d, theta).

    A_i =
    [[ cosθ, -sinθ cosα,  sinθ sinα,  a cosθ],
     [ sinθ,  cosθ cosα, -cosθ sinα,  a sinθ],
     [    0,        sinα,       cosα,      d],
     [    0,           0,          0,      1]]
    """
    a: float
    alpha: float
    d: float
    theta: float

    def T(self) -> SE3:
        ca, sa = math.cos(self.alpha), math.sin(self.alpha)
        ct, st = math.cos(self.theta), math.sin(self.theta)
        R = np.array(
            [[ct, -st * ca,  st * sa],
             [st,  ct * ca, -ct * sa],
             [0.0,     sa,       ca]],
            dtype=float,
        )
        t = np.array([self.a * ct, self.a * st, self.d], dtype=float)
        return SE3(R, t)


@dataclass
class KinematicChain:
    """
    Product of DH links (forward kinematics).
    """
    links: Tuple[DHLink, ...] | List[DHLink]

    def fk(self) -> SE3:
        T = SE3.identity()
        for link in self.links:
            T = T @ link.T()
        return T
