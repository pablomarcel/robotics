# motion_kinematics/apis.py
"""
Public computation facade for Motion Kinematics.

The `APIs` class is a thin, object-oriented facade that groups core
computations used by the application and CLI. Every method is pure (no I/O),
returns testable payloads (dicts with numpy arrays and metadata), and keeps the
math close to the relevant equations for traceability.

Covered topics (non-exhaustive):
- Axis–angle rotations (Rodrigues)  ➜ eqs. 3.187, used widely through §4
- Homogeneous transforms (SE(3))    ➜ eqs. 4.59–4.66, 4.75–4.90
- Screw motions (general/central)   ➜ eqs. 4.206, 4.220–4.222, 4.279–4.283
- Plücker lines, angle & distance   ➜ eqs. 4.344–4.346, 4.389–4.390
- Plane/point distance              ➜ eqs. 4.391–4.397 (with s = 0 default)
- Forward kinematics (DH)           ➜ standard DH product for §4.4, §4.5

Notes
-----
• This module depends only on NumPy for numerics to keep tests fast/stable.
• Results are returned in a JSON-serializable shape (NumPy arrays are OK and
  will be converted by the JSON encoder in `app.py` when persisted).
• If you later split math into `core.py` services, you can keep the public API
  stable and just delegate there (minimal refactor).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import math
import numpy as np


# ------------------------------ helpers --------------------------------------
def _as_vec3(v: Tuple[float, float, float]) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError("Vector has non-finite entries")
    return a


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector cannot be normalized")
    return v / n


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix [v]_x."""
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=float)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rodrigues rotation_kinematics (eq. 3.187), used throughout Chapter 4.
    R = I + sin(phi)[u]_x + (1 - cos(phi))[u]_x^2
    """
    u = _normalize(axis)
    K = _skew(u)
    s, c = math.sin(angle), math.cos(angle)
    R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
    return R


def _hom(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4×4 homogeneous transform from (R, t)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


# ------------------------------ facade ---------------------------------------
@dataclass
class APIs:
    """
    A stateless facade grouping kinematics operations.

    All methods return a dict with:
      - 'matrix' / 'R' / 't' / 'T' / 'meta': numpy arrays and metadata
    """

    # --------------------------- Rotations --------------------------------
    def rotation_axis_angle(
        self,
        axis: Tuple[float, float, float],
        angle: float,
        *,
        degrees: bool = False,
    ) -> Dict[str, object]:
        """
        Build a 3×3 rotation_kinematics using Rodrigues (eq. 3.187).

        Parameters
        ----------
        axis : 3-tuple
            Rotation axis (need not be normalized).
        angle : float
            Rotation angle.
        degrees : bool
            If True, `angle` is in degrees.

        Returns
        -------
        dict with keys: 'R', 'axis', 'angle'
        """
        a = _as_vec3(axis)
        phi = math.radians(angle) if degrees else float(angle)
        R = _rodrigues(a, phi)
        return {"R": R, "axis": _normalize(a), "angle": phi, "meta": {"degrees": degrees}}

    # --------------------------- Screw motions ----------------------------
    def screw_motion(
        self,
        u: Tuple[float, float, float],
        s: Tuple[float, float, float],
        h: float,
        phi: float,
        *,
        degrees: bool = False,
    ) -> Dict[str, object]:
        """
        General screw motion_kinematics homogeneous transform (eqs. 4.206 and 4.220–4.222).

        Using unit axis û, location vector s, pitch h and rotation_kinematics φ:
            R = I cosφ + û ûᵀ (1 - cosφ) + [û]ₓ sinφ  (Rodrigues, 4.221)
            g_d = s - R s + h û                           (4.220 / 4.222)
            T = [[R, g_d],
                 [0,   1 ]]

        Returns dict with keys: 'T', 'R', 't', 'axis', 's', 'h', 'phi'
        """
        u_hat = _normalize(_as_vec3(u))
        s_vec = _as_vec3(s)
        ang = math.radians(phi) if degrees else float(phi)
        R = _rodrigues(u_hat, ang)
        t = (s_vec - R @ s_vec) + (h * u_hat)
        T = _hom(R, t)
        return {
            "T": T, "R": R, "t": t,
            "axis": u_hat, "s": s_vec, "h": float(h), "phi": ang,
            "meta": {"degrees": degrees, "eq": [4.206, 4.220, 4.221, 4.222]},
        }

    # --------------------------- Plücker lines ----------------------------
    def plucker_from_points(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
    ) -> Dict[str, object]:
        """
        Build Plücker coordinates from two points (eqs. 4.344–4.346).

        û = (p2 - p1)/||p2 - p1||,   ρ = p1 × û,    constraint û·ρ = 0
        """
        r1 = _as_vec3(p1)
        r2 = _as_vec3(p2)
        u_hat = _normalize(r2 - r1)
        rho = np.cross(r1, u_hat)
        return {
            "u": u_hat, "rho": rho,
            "l": np.concatenate([u_hat, rho]),
            "constraints": {"orthogonality": float(u_hat @ rho)},
            "meta": {"eq": [4.344, 4.346]},
        }

    def plucker_angle_distance(
        self,
        a1: Tuple[float, float, float],
        a2: Tuple[float, float, float],
        b1: Tuple[float, float, float],
        b2: Tuple[float, float, float],
    ) -> Dict[str, object]:
        """
        Angle (4.389) and shortest distance (4.390) between two lines.

        Given l1 = [û1, ρ1], l2 = [û2, ρ2]:
            sin α = ||û1 × û2||                             (4.389)
            d = |û2·ρ1 + û1·ρ2| / sin α,  for non-parallel lines  (4.390)
        """
        L1 = self.plucker_from_points(a1, a2)
        L2 = self.plucker_from_points(b1, b2)
        u1, rho1 = L1["u"], L1["rho"]
        u2, rho2 = L2["u"], L2["rho"]

        cross = np.linalg.norm(np.cross(u1, u2))
        # Clamp for numerical safety
        cross = float(np.clip(cross, 0.0, 1.0))
        alpha = math.asin(cross)
        # Distance formula (handle parallel as NaN)
        num = float(u2 @ rho1 + u1 @ rho2)
        d = float(abs(num) / cross) if cross > 1e-12 else float("nan")

        return {
            "alpha": alpha,
            "sin_alpha": cross,
            "distance": d,
            "lines": {
                "l1": {"u": u1, "rho": rho1},
                "l2": {"u": u2, "rho": rho2},
            },
            "meta": {"eq": [4.389, 4.390]},
        }

    # ------------------------------ Planes --------------------------------
    def plane_point_distance(
        self,
        point: Tuple[float, float, float],
        normal: Tuple[float, float, float],
        *,
        s: float = 0.0,
        signed: bool = True,
    ) -> Dict[str, object]:
        """
        Distance from a point to a plane (eqs. 4.391–4.397).

        Plane: n̂·x = s  (4.391), where s is the minimum distance of the plane
        to the origin. By default we assume s = 0 (plane through origin).

        The signed distance is n̂·p - s. The unsigned distance is its absolute value.
        """
        n_hat = _normalize(_as_vec3(normal))
        p = _as_vec3(point)
        d_signed = float(n_hat @ p - float(s))
        d = d_signed if signed else abs(d_signed)
        return {
            "distance": d,
            "signed": d_signed,
            "normal": n_hat,
            "s": float(s),
            "point": p,
            "meta": {"eq": [4.391, 4.397]},
        }

    # ----------------------- Forward Kinematics (DH) ----------------------
    def forward_kinematics(
        self,
        dh_params: Iterable[Iterable[float]],
    ) -> Dict[str, object]:
        """
        Standard Denavit–Hartenberg forward_kinematics kinematics (product of Aᵢ matrices).

        Each row is [a_i, alpha_i, d_i, theta_i] in radians.  The individual link
        transform A_i is:

            A_i =
            [[ cosθ, -sinθ cosα,  sinθ sinα,  a cosθ],
             [ sinθ,  cosθ cosα, -cosθ sinα,  a sinθ],
             [  0,        sinα,       cosα,        d],
             [  0,          0,         0,         1]]

        Returns
        -------
        dict with keys: 'T', 'links' (list of A_i), 'cumulative' (prefix products)
        """
        A_mats: List[np.ndarray] = []
        cumulative: List[np.ndarray] = []

        T = np.eye(4)
        for row in dh_params:
            a, alpha, d, theta = map(float, row)
            ca, sa = math.cos(alpha), math.sin(alpha)
            ct, st = math.cos(theta), math.sin(theta)

            A = np.array(
                [
                    [ct, -st * ca, st * sa, a * ct],
                    [st, ct * ca, -ct * sa, a * st],
                    [0.0, sa, ca, d],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            A_mats.append(A)
            T = T @ A
            cumulative.append(T.copy())

        return {"T": T, "links": A_mats, "cumulative": cumulative, "meta": {"convention": "DH"}}
