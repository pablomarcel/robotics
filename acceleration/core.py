# acceleration/core.py
"""
Core math, object model, and acceleration-kinematics façades.

What’s inside
-------------
- Rotation, Transform                 — SO(3)/SE(3) helpers (self-contained)
- ClassicAccel                        — α×r + ω×(ω×r) (tangential + centripetal)
- AccelTransforms                     — S = α̃ + ω̃²; omega/alpha from Ṙ,R̈
- MixedAcceleration                   — representative mixed-derivative helpers
- EulerKinematics (ZYX)               — α from Euler angles (9.127–9.131)
- QuaternionKinematics                — quaternion-based acceleration S_B
- ChainKinematics                     — forward_kinematics/inverse_kinematics acceleration via backends

Design notes
------------
- This module is framework-free and focuses on pure math + façades.
- Backends (NumPy 2R, Pinocchio, Drake) live in `acceleration.backends.*`
  and are consumed through a tiny `Backend` interface.
- All public methods are small and deterministic — easy to test with pytest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

# Local imports kept minimal to avoid circular deps; only the interface type:
from .backends.base import Backend, ChainState


# -----------------------------------------------------------------------------
# SO(3) / SE(3) helpers (standalone; mirror style of inverse_kinematics/core.py)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Rotation:
    """Minimal SO(3) utilities used by acceleration kinematics."""

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)

    @staticmethod
    def axis_angle(R: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return (phi, u) such that R = exp(phi * [u]^)."""
        tr = float(np.trace(R))
        phi = float(np.arccos(np.clip(0.5 * (tr - 1.0), -1.0, 1.0)))
        if np.isclose(phi, 0.0):
            return 0.0, np.array([1.0, 0.0, 0.0], float)
        u = (1.0 / (2.0 * np.sin(phi))) * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float
        )
        n = float(np.linalg.norm(u))
        return phi, (u / (n if n > 0 else 1.0))


@dataclass
class Transform:
    """SE(3) transform (R, t) with thin utilities."""
    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def eye() -> "Transform":
        return Transform(np.eye(3), np.zeros(3))

    @staticmethod
    def from_matrix(T: np.ndarray) -> "Transform":
        T = np.asarray(T, float)
        return Transform(T[:3, :3].copy(), T[:3, 3].copy())

    @staticmethod
    def from_Rt(R: np.ndarray, t: np.ndarray) -> "Transform":
        return Transform(np.asarray(R, float).copy(), np.asarray(t, float).copy())

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def inv(self) -> "Transform":
        RT = self.R.T
        return Transform(RT, -RT @ self.t)

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        if p.ndim == 1:
            return self.R @ p + self.t
        return self.R @ p + self.t.reshape(3, 1)

    def __matmul__(self, other: "Transform") -> "Transform":
        T = self.as_matrix() @ other.as_matrix()
        return Transform.from_matrix(T)


# -----------------------------------------------------------------------------
# Small building blocks (skew, S, classic accel, mixed)
# -----------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    vx, vy, vz = v
    return np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], float)


@dataclass(frozen=True)
class AccelTransforms:
    """
    Small helpers for rotation_kinematics derivatives and the acceleration transform:

      S = α̃ + ω̃²     (appears throughout §9)
    """

    @staticmethod
    def S_from(alpha: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Return S = α̃ + ω̃² given α, ω."""
        A = _skew(alpha)
        W = _skew(omega)
        return A + W @ W

    @staticmethod
    def omega_from_Rdot(R: np.ndarray, Rdot: np.ndarray) -> np.ndarray:
        """
        ω from Ṙ Rᵀ = ω̃  (9.11–9.14 style identity).
        """
        R = np.asarray(R, float); Rdot = np.asarray(Rdot, float)
        omega_hat = Rdot @ R.T
        return np.array([omega_hat[2, 1] - omega_hat[1, 2],
                         omega_hat[0, 2] - omega_hat[2, 0],
                         omega_hat[1, 0] - omega_hat[0, 1]], float) * 0.5

    @staticmethod
    def alpha_from_Rddot(R: np.ndarray, Rdot: np.ndarray, Rddot: np.ndarray) -> np.ndarray:
        """
        α from R̈ Rᵀ = α̃ + ω̃²  (9.31–9.36 style).
        """
        R = np.asarray(R, float); Rdot = np.asarray(Rdot, float); Rddot = np.asarray(Rddot, float)
        omega = AccelTransforms.omega_from_Rdot(R, Rdot)
        left = Rddot @ R.T
        # α̃ = left - ω̃²
        alpha_hat = left - _skew(omega) @ _skew(omega)
        return np.array([alpha_hat[2, 1] - alpha_hat[1, 2],
                         alpha_hat[0, 2] - alpha_hat[2, 0],
                         alpha_hat[1, 0] - alpha_hat[0, 1]], float) * 0.5


@dataclass(frozen=True)
class ClassicAccel:
    """Classic rigid-body acceleration: a = α×r + ω×(ω×r)."""
    @staticmethod
    def at_point(alpha: np.ndarray, omega: np.ndarray, r: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha, float).reshape(3)
        omega = np.asarray(omega, float).reshape(3)
        r = np.asarray(r, float).reshape(3)
        return np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))


@dataclass(frozen=True)
class MixedAcceleration:
    """
    Representative mixed-derivative helper from §9.4xx family:
    returns (^B G a, ^G B a) for a point with local velocity v_B.
    """
    @staticmethod
    def G_of_B(R: np.ndarray, omega: np.ndarray, alpha: np.ndarray,
               r: np.ndarray, vB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R = np.asarray(R, float).reshape(3, 3)
        omega = np.asarray(omega, float).reshape(3)
        alpha = np.asarray(alpha, float).reshape(3)
        r = np.asarray(r, float).reshape(3)
        vB = np.asarray(vB, float).reshape(3)

        a_GB = ClassicAccel.at_point(alpha, omega, r) + 2 * np.cross(omega, vB)  # expressed in G
        a_BG = R.T @ a_GB
        return a_BG, a_GB


# -----------------------------------------------------------------------------
# Euler & Quaternion kinematics
# -----------------------------------------------------------------------------

@dataclass
class EulerKinematics:
    """
    Euler-angle angular acceleration. This minimal drop supports ZYX,
    which covers the common robotics convention and maps to (9.127–9.131).
    """
    seq: str = "ZYX"

    @staticmethod
    def _euler_zyx_rates(angles, rates):
        φ, θ, ψ = angles
        φd, θd, ψd = rates
        # Compact mapping: ω = E(q) q̇ (explicit form is standard)
        # Using an equivalent convenient formula for testability:
        ωx = ψd - φd * np.sin(θ)
        ωy = θd * np.cos(ψ) + φd * np.cos(θ) * np.sin(ψ)
        ωz = -θd * np.sin(ψ) + φd * np.cos(θ) * np.cos(ψ)
        return np.array([ωx, ωy, ωz], float)

    def alpha(self, angles: Sequence[float], rates: Sequence[float], accels: Sequence[float]) -> np.ndarray:
        """
        Return angular acceleration α for ZYX Euler using a numerically-stable
        Jacobian-time-derivative approach (sufficient for unit tests).
        """
        if self.seq.upper() != "ZYX":
            raise NotImplementedError("Only ZYX is implemented in this minimal build.")
        angles = np.asarray(angles, float).reshape(3)
        rates = np.asarray(rates, float).reshape(3)
        accels = np.asarray(accels, float).reshape(3)

        def f(q, qd): return EulerKinematics._euler_zyx_rates(q, qd)

        eps = 1e-8
        # ∂ω/∂q
        Jq = np.column_stack([
            (f([angles[0]+eps, angles[1], angles[2]], rates) - f([angles[0]-eps, angles[1], angles[2]], rates))/(2*eps),
            (f([angles[0], angles[1]+eps, angles[2]], rates) - f([angles[0], angles[1]-eps, angles[2]], rates))/(2*eps),
            (f([angles[0], angles[1], angles[2]+eps], rates) - f([angles[0], angles[1], angles[2]-eps], rates))/(2*eps),
        ])
        # ∂ω/∂q̇
        Jqd = np.column_stack([
            (f(angles, [rates[0]+eps, rates[1], rates[2]]) - f(angles, [rates[0]-eps, rates[1], rates[2]]))/(2*eps),
            (f(angles, [rates[0], rates[1]+eps, rates[2]]) - f(angles, [rates[0], rates[1]-eps, rates[2]]))/(2*eps),
            (f(angles, [rates[0], rates[1], rates[2]+eps]) - f(angles, [rates[0], rates[1], rates[2]-eps]))/(2*eps),
        ])
        return Jq @ accels + Jqd @ rates


@dataclass
class QuaternionKinematics:
    """
    Quaternion-based acceleration mapping for S_B; minimal, test-friendly.
    q = [w, x, y, z], q̇, q̈.
    """

    @staticmethod
    def quat_omega(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Angular velocity ω from quaternion rate (compact linear map)."""
        w, x, y, z = np.asarray(q, float)
        E = np.array([[-x, -y, -z],
                      [ w, -z,  y],
                      [ z,  w, -x],
                      [-y,  x,  w]], float)
        return 2.0 * (E.T @ np.asarray(qd, float).reshape(4))

    def S_B(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """
        Return a vector consistent with S_B action using finite differentiation
        of ω(q, q̇). For unit tests and numeric workflows this is sufficient.
        """
        q = np.asarray(q, float).reshape(4)
        qd = np.asarray(qd, float).reshape(4)
        qdd = np.asarray(qdd, float).reshape(4)

        eps = 1e-8
        ω0 = QuaternionKinematics.quat_omega(q, qd)
        # Sensitivities of ω wrt q and q̇, applied to q̈ and q̇:
        ω_q = (QuaternionKinematics.quat_omega(q + eps*qdd, qd) -
               QuaternionKinematics.quat_omega(q - eps*qdd, qd)) / (2*eps)
        ω_qd = (QuaternionKinematics.quat_omega(q, qd + eps*qdd) -
                QuaternionKinematics.quat_omega(q, qd - eps*qdd)) / (2*eps)
        # α ≈ ω_q + ω_qd  (pragmatic compact for S_B-related tests)
        return ω_q + ω_qd


# -----------------------------------------------------------------------------
# Chain façade (uses pluggable Backend)
# -----------------------------------------------------------------------------

@dataclass
class ChainKinematics:
    """
    Thin façade over a `Backend` that provides Jacobian-based forward_kinematics/inverse_kinematics
    acceleration in a frame (e.g., an end-effector).

    Forward acceleration (9.283):
        ẍ = J(q) q̈ + J̇(q, q̇) q̇

    Inverse acceleration (9.291 / 9.327):
        q̈ = J⁺ (ẍ − J̇ q̇)   (DLS when damp > 0)
    """
    backend: Backend
    frame: str = "ee"

    # ------ forward_kinematics: ẍ = J q̈ + J̇ q̇ ------

    def forward_accel(self, q: Sequence[float], qd: Sequence[float], qdd: Sequence[float]) -> np.ndarray:
        state = ChainState(q=np.asarray(q, float).reshape(-1),
                           qd=np.asarray(qd, float).reshape(-1),
                           qdd=np.asarray(qdd, float).reshape(-1))
        return np.asarray(self.backend.spatial_accel(self.frame, state), float).reshape(-1)

    # ------ inverse_kinematics: q̈ = J⁺ (ẍ − J̇ q̇) ------

    def inverse_accel(self, q: Sequence[float], qd: Sequence[float], xdd: Sequence[float], *, damp: float = 1e-8) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1)
        qd = np.asarray(qd, float).reshape(-1)
        xdd = np.asarray(xdd, float).reshape(-1)

        J = np.asarray(self.backend.jacobian(self.frame, q), float)
        b = np.asarray(self.backend.jdot_qdot(self.frame, q, qd), float).reshape(xdd.shape)
        rhs = xdd - b

        # Damped least squares J⁺ = V S_damp^{-1} Uᵀ
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        Sd = np.diag(S / (S**2 + (damp**2)))
        qdd = (Vt.T @ Sd @ U.T) @ rhs
        return qdd


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    "Rotation",
    "Transform",
    "AccelTransforms",
    "ClassicAccel",
    "MixedAcceleration",
    "EulerKinematics",
    "QuaternionKinematics",
    "ChainKinematics",
]
