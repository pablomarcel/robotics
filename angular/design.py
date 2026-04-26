# angular/design.py
"""
Preset builders for **angular-kinematics** workflows.

These presets construct small, testable OOP models for common scenarios used to
exercise Eq. 7.1–7.416:

- Pure spin about a fixed axis (Rodrigues, (7.21–7.23))
- Constant screw motion (twist decomposition, (7.287–7.292), small-twist dT (7.367/7.373))
- Rigid body with moving origin (velocity matrix & transport theorem, (7.264–7.269), (7.351))
- Spherical wrist (3R) angular-rate mapping (sum of rotating axes, (7.295–7.303), (7.319–7.320))
- Planar 2R tip velocity (illustrative 7.325–7.347 block; ω_z and v in space)

All presets are thin OOP wrappers around the kernels in :mod:`angular.core` and
return NumPy arrays or instances of :class:`angular.core.Rotation`,
:class:`angular.core.Transform`, or convenience outputs (ω, V, etc.), so you can
plug them directly into tests, the CLI façade, or notebooks.

Design notes
------------
* We keep these builders **numeric** and dependency-light (no SymPy).
* Where an instantaneous mapping is well-known (e.g., spherical wrist), we
  compute **space-frame axes** and sum `ω = Σ ai(q) * qd_i`.
* For 2R planar, we provide the tip linear velocity in the space frame and the
  scalar out-of-plane ω_z, matching the block structure in 7.325–7.347.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .core import Rotation, Transform, KinematicsEngine, Screw


# ------------------------------ Utilities --------------------------------- #

def _Rz(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]])

def _Rx(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  ca, -sa],
                     [0.0,  sa,  ca]])

def _Ry(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ ca, 0.0, sa],
                     [0.0, 1.0, 0.0],
                     [-sa, 0.0, ca]])


# ------------------------------- Pure spin -------------------------------- #

@dataclass(frozen=True)
class PureSpin:
    """
    **Pure rotation_kinematics** about a fixed unit axis `u` with scalar rate `phi_dot`.

    Provides:
    - SO(3) via Rodrigues (7.115–7.117)
    - ω and ω~ from (7.21–7.23)

    Parameters
    ----------
    u : Sequence[float]
        Unit rotation_kinematics axis (3,).
    phi : float
        Rotation angle (rad).
    phi_dot : float
        Instantaneous angular rate (rad/s).
    """
    u: Sequence[float]
    phi: float
    phi_dot: float

    def rotation(self) -> Rotation:
        u = np.asarray(self.u, float)
        return Rotation.from_axis_angle(u, self.phi)

    def omega(self) -> np.ndarray:
        u = np.asarray(self.u, float) / (np.linalg.norm(self.u) + 1e-12)
        return self.phi_dot * u

    def Rdot(self) -> np.ndarray:
        R = self.rotation().R
        return Rotation(R).Rdot_from_omega(self.omega())


# ----------------------------- Constant screw ----------------------------- #

@dataclass(frozen=True)
class ConstantScrewMotion:
    """
    **Constant screw motion** described by a spatial twist V = [ω; v].

    Provides:
    - Screw decomposition (axis `s`, moment `m`, pitch `p`) per (7.287–7.292)
    - Small-twist differential transform `dT ≈ [I + dD][I + dR]` (7.367/7.373)

    Parameters
    ----------
    twist : Sequence[float]
        6-vector spatial twist [ωx, ωy, ωz, vx, vy, vz] expressed in space frame.
    """
    twist: Sequence[float]

    def screw(self) -> Screw:
        V = np.asarray(self.twist, float).reshape(6)
        return Screw.from_twist(V)

    def differential_transform(self, dt: float) -> np.ndarray:
        """
        Small-step differential transform for timestep `dt`.
        """
        V = np.asarray(self.twist, float).reshape(6)
        w, v = V[:3], V[3:]
        dphiu = w * dt
        dd = v * dt
        # Use KinematicsEngine kernel; returns (T - I), so add I here for dT
        dT_minus_I = KinematicsEngine.differential_transform(dphiu, dd)
        return np.eye(4) + dT_minus_I


# ------------------- Rigid body with moving origin (SE(3)) ---------------- #

@dataclass(frozen=True)
class MovingOriginRigidBody:
    """
    Rigid body B with space-frame origin trajectory d_B^G(t), velocity ḋ_B^G,
    and angular velocity ω^G. Computes point velocity per (7.264–7.269) and
    the velocity matrix V = Ṫ T^{-1} (7.351).

    Parameters
    ----------
    R : np.ndarray
        Rotation ^G R_B (3×3).
    d : np.ndarray
        Origin position d_B^G (3,).
    omega_g : np.ndarray
        Angular velocity ^G ω_B (3,).
    d_dot : np.ndarray
        Origin linear velocity ḋ_B^G (3,).
    """
    R: np.ndarray
    d: np.ndarray
    omega_g: np.ndarray
    d_dot: np.ndarray

    def transform(self) -> Transform:
        return Transform(self.R, self.d)

    def point_velocity(self, r_gp: np.ndarray) -> np.ndarray:
        """
        v_P^G = ω^G × (r_GP - d_B^G) + ḋ_B^G  (Eq. 7.264–7.269).
        """
        return KinematicsEngine.rigid_body_point_velocity(self.omega_g, r_gp, self.d, self.d_dot)

    def velocity_matrix(self, Rdot: Optional[np.ndarray] = None) -> np.ndarray:
        """
        V = Ṫ T^{-1} = [[ω~, v], [0,0]], with v = ḋ - ω~ d  (Eq. 7.351).
        If `Rdot` not provided, uses ω~ R for Ṙ.
        """
        if Rdot is None:
            Rdot = Rotation(self.R).Rdot_from_omega(self.omega_g)
        return Transform(self.R, self.d).velocity_matrix(Rdot, self.d_dot)


# ------------------------------ Spherical wrist --------------------------- #

@dataclass(frozen=True)
class SphericalWrist:
    """
    Ideal **3R spherical wrist** with joint axes intersecting at the wrist center.

    We support three effective rotation_kinematics patterns via local-axis choices:
        type 1: Z–X–Z
        type 2: Z–Y–Z
        type 3: X–Y–Z

    The instantaneous space-frame wrist angular velocity is
        ω^G(q, q̇) = a1(q) q̇1 + a2(q) q̇2 + a3(q) q̇3,
    where `ai(q)` are the current joint axes expressed in the space frame.

    Parameters
    ----------
    wrist_type : {1, 2, 3}
        Rotation-axis pattern (see above).
    """
    wrist_type: int = 1  # 1: Z-X-Z, 2: Z-Y-Z, 3: X-Y-Z

    def _axes_space(self, q1: float, q2: float, q3: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the three instantaneous joint axes expressed in the space frame.
        """
        if self.wrist_type == 1:           # Z, then X (rotated by Rz(q1)), then Z (by Rz(q1)Rx(q2))
            a1 = np.array([0., 0., 1.])
            a2 = _Rz(q1) @ np.array([1., 0., 0.])
            a3 = (_Rz(q1) @ _Rx(q2)) @ np.array([0., 0., 1.])
        elif self.wrist_type == 2:         # Z, then Y, then Z
            a1 = np.array([0., 0., 1.])
            a2 = _Rz(q1) @ np.array([0., 1., 0.])
            a3 = (_Rz(q1) @ _Ry(q2)) @ np.array([0., 0., 1.])
        elif self.wrist_type == 3:         # X, then Y (rotated by Rx(q1)), then Z (by Rx(q1)Ry(q2))
            a1 = np.array([1., 0., 0.])
            a2 = _Rx(q1) @ np.array([0., 1., 0.])
            a3 = (_Rx(q1) @ _Ry(q2)) @ np.array([0., 0., 1.])
        else:
            raise ValueError("wrist_type must be in {1,2,3}")
        return a1, a2, a3

    def omega(self, q: Sequence[float], qd: Sequence[float]) -> np.ndarray:
        """
        ω^G = a1(q) q̇1 + a2(q) q̇2 + a3(q) q̇3  (sum of rotating axes).
        """
        q1, q2, q3 = [float(x) for x in q]
        qd1, qd2, qd3 = [float(x) for x in qd]
        a1, a2, a3 = self._axes_space(q1, q2, q3)
        return qd1 * a1 + qd2 * a2 + qd3 * a3

    def axes_space(self, q: Sequence[float]) -> np.ndarray:
        """
        Return 3×3 matrix A(q) whose columns are the current axes [a1, a2, a3].
        This is the wrist angular-velocity Jacobian (space-frame) for unit rates.
        """
        q1, q2, q3 = [float(x) for x in q]
        a1, a2, a3 = self._axes_space(q1, q2, q3)
        return np.column_stack([a1, a2, a3])


# ------------------------------- Planar 2R -------------------------------- #

@dataclass(frozen=True)
class Planar2RVelocity:
    """
    **Planar 2R** end-effector instantaneous velocity in the space frame.

    With link lengths (l1, l2) and joint angles (q1, q2):
      x = l1 cos q1 + l2 cos(q1+q2)
      y = l1 sin q1 + l2 sin(q1+q2)
    The tip linear velocity v^G = J(q) q̇ with
      J = [[-l1 sin q1 - l2 sin(q1+q2),  -l2 sin(q1+q2)],
           [ l1 cos q1 + l2 cos(q1+q2),   l2 cos(q1+q2)]]
    and out-of-plane ω_z = q̇1 + q̇2.

    This corresponds to the illustrative velocity-matrix blocks in (7.325–7.347).

    Parameters
    ----------
    l1 : float
        First link length.
    l2 : float
        Second link length.
    """
    l1: float
    l2: float

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        q1, q2 = float(q[0]), float(q[1])
        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
        J = np.array([
            [-self.l1 * s1 - self.l2 * s12,  -self.l2 * s12],
            [ self.l1 * c1 + self.l2 * c12,   self.l2 * c12]
        ])
        return J

    def linear_velocity(self, q: Sequence[float], qd: Sequence[float]) -> np.ndarray:
        J = self.jacobian(q)
        qd = np.asarray(qd, float).reshape(2)
        return J @ qd

    def omega_z(self, qd: Sequence[float]) -> float:
        qd1, qd2 = float(qd[0]), float(qd[1])
        return qd1 + qd2


# --------------------------- Convenience Builders ------------------------- #

def pure_spin_about(axis: Sequence[float], phi: float, phi_dot: float) -> PureSpin:
    """
    Convenience factory for a **PureSpin** scenario.
    """
    return PureSpin(axis, phi, phi_dot)


def constant_screw(twist6: Sequence[float]) -> ConstantScrewMotion:
    """
    Convenience factory for a **ConstantScrewMotion** scenario.
    """
    return ConstantScrewMotion(twist6)


def moving_origin_rigidbody(R: np.ndarray, d: np.ndarray,
                            omega_g: np.ndarray, d_dot: np.ndarray) -> MovingOriginRigidBody:
    """
    Convenience factory for a **MovingOriginRigidBody** scenario.
    """
    return MovingOriginRigidBody(R, d, omega_g, d_dot)


def spherical_wrist(*, wrist_type: int = 1) -> SphericalWrist:
    """
    Convenience factory for a **SphericalWrist** scenario.
    """
    return SphericalWrist(wrist_type=wrist_type)


def planar_2r(l1: float, l2: float) -> Planar2RVelocity:
    """
    Convenience factory for a **Planar2RVelocity** scenario.
    """
    return Planar2RVelocity(l1=float(l1), l2=float(l2))
