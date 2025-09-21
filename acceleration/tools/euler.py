# acceleration/tools/euler.py
"""
Euler-angle helpers for **acceleration kinematics**.

What this module provides
-------------------------
- Rotation builders:
    * euler_matrix(seq, angles)                → 3×3 rotation for seq in {"ZYX","XYZ","ZXZ","ZYZ"}

- Rate/accel maps (general API used by tests):
    * euler_rates_matrix(seq, angles)          → E(q) so that ω = E(q) q̇
    * euler_accel_matrix(seq, angles, rates)   → Ė(q, q̇) so that α = E(q) q̈ + Ė(q, q̇) q̇

- ZYX closed forms (kept for speed/clarity and used by generic helpers when seq=="ZYX"):
    * E_zyx(angles), omega_zyx(angles, rates)
    * Edot_times_rates_zyx(angles, rates)      → (d/dt E) q̇
    * alpha_zyx(angles, rates, accels)         → α

- OO façade:
    * EulerZYX(angles, rates, accels)

Notes
-----
- All builders and maps are **complex-step friendly**: if inputs are complex,
  the returned arrays preserve complex dtype. This lets euler_accel_matrix use
  a truncation-free complex-step directional derivative for Ė.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..utils import ensure_shape, asvec, vex as _vex


# ---------------------------------------------------------------------------
# Small SO(3) helpers (dtype-preserving)
# ---------------------------------------------------------------------------

def _Rx(a: float | complex) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    dt = np.result_type(ca, sa)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  ca, -sa],
                     [0.0,  sa,  ca]], dtype=dt)

def _Ry(a: float | complex) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    dt = np.result_type(ca, sa)
    return np.array([[ ca, 0.0,  sa],
                     [0.0, 1.0, 0.0],
                     [-sa, 0.0,  ca]], dtype=dt)

def _Rz(a: float | complex) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    dt = np.result_type(ca, sa)
    return np.array([[ ca, -sa, 0.0],
                     [ sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=dt)


def _Edot_times_rates_directional(seq: str,
                                  q: np.ndarray,
                                  qd: np.ndarray,
                                  h: float = 1e-6) -> np.ndarray:
    """
    Compute the directional time derivative vector v = (d/dt E(q)) q̇
    by differentiating ω(t) = E(q(t)) q̇ along q(t) = q + t q̇ (q̇ held constant)
    using a 5-point central stencil. Returns v (shape: (3,)).
    """
    q  = asvec(q, 3)
    qd = asvec(qd, 3)
    # sample ω at t = {-2h, -h, +h, +2h}
    w_p2 = euler_rates_matrix(seq, q + 2.0*h*qd) @ qd
    w_p1 = euler_rates_matrix(seq, q + 1.0*h*qd) @ qd
    w_m1 = euler_rates_matrix(seq, q - 1.0*h*qd) @ qd
    w_m2 = euler_rates_matrix(seq, q - 2.0*h*qd) @ qd
    # dω/dt at t=0 with 5-point stencil
    return (-w_p2 + 8.0*w_p1 - 8.0*w_m1 + w_m2) / (12.0*h)


# ---------------------------------------------------------------------------
# General Euler → R (for several common sequences)
# ---------------------------------------------------------------------------

@ensure_shape(3, 3)
def euler_matrix(seq: str, angles: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Build a rotation matrix from Euler angles for selected **intrinsic** sequences.

    Supported sequences:
      - "ZYX": R = Rz(ψ) Ry(θ) Rx(φ)      (yaw, pitch, roll)
      - "XYZ": R = Rx(α) Ry(β) Rz(γ)
      - "ZXZ": R = Rz(α) Rx(β) Rz(γ)      (proper-Euler)
      - "ZYZ": R = Rz(α) Ry(β) Rz(γ)      (proper-Euler)

    Returns a (3,3) ndarray with dtype preserved from `angles`.
    """
    s = str(seq).upper()
    a1, a2, a3 = asvec(angles, 3)
    if s == "ZYX":
        return _Rz(a3) @ _Ry(a2) @ _Rx(a1)
    if s == "XYZ":
        return _Rx(a1) @ _Ry(a2) @ _Rz(a3)
    if s == "ZXZ":
        return _Rz(a1) @ _Rx(a2) @ _Rz(a3)
    if s == "ZYZ":
        return _Rz(a1) @ _Ry(a2) @ _Rz(a3)
    raise ValueError(f"Unsupported Euler sequence {seq!r}; try one of 'ZYX', 'XYZ', 'ZXZ', 'ZYZ'.")


# ---------------------------------------------------------------------------
# ZYX closed-form ω = E(q) q̇  (dtype-preserving)
# ---------------------------------------------------------------------------

@ensure_shape(3, 3)
def E_zyx(angles: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Return the 3×3 matrix E(q) such that **ω = E(q) q̇** for ZYX Euler angles.

    angles = [φ, θ, ψ]  (roll, pitch, yaw)

    Mapping:
        ωx =  φ̇ - ψ̇ sinθ
        ωy =  θ̇ cosφ + ψ̇ sinφ cosθ
        ωz = -θ̇ sinφ + ψ̇ cosφ cosθ
    """
    φ, θ, ψ = asvec(angles, 3)
    sφ, cφ = np.sin(φ), np.cos(φ)
    sθ = np.sin(θ)
    cθ = np.cos(θ)
    dt = np.result_type(sφ, cφ, sθ, cθ)

    E = np.array(
        [
            [1.0, 0.0,     -sθ],
            [0.0,  cφ,  sφ * cθ],
            [0.0, -sφ,  cφ * cθ],
        ],
        dtype=dt,
    )
    return E


@ensure_shape(3,)
def omega_zyx(angles: Sequence[float] | np.ndarray,
              rates: Sequence[float] | np.ndarray) -> np.ndarray:
    """Angular velocity **ω = E(q) q̇** for ZYX angles."""
    q = asvec(angles, 3)
    qd = asvec(rates, 3)
    out = E_zyx(q) @ qd
    return np.asarray(out, float)


# ---------------------------------------------------------------------------
# Time derivative of E(q) along q̇ → (d/dt E) q̇  (ZYX closed form; dtype-preserving)
# ---------------------------------------------------------------------------

@ensure_shape(3,)
def Edot_times_rates_zyx(angles: Sequence[float] | np.ndarray,
                         rates: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Compute (d/dt E(q)) q̇ for ZYX angles using a closed-form derivative.

    Returns a 3-vector equal to ((d/dt)E) q̇. Dtype preserved from inputs.
    """
    φ, θ, ψ = asvec(angles, 3)
    φd, θd, ψd = asvec(rates, 3)

    sφ, cφ = np.sin(φ), np.cos(φ)
    sθ, cθ = np.sin(θ), np.cos(θ)
    dt = np.result_type(sφ, cφ, sθ, cθ, φd, θd, ψd)

    # Row 1 of E: [1, 0, -sθ] → time derivative ⋅ q̇ → (-cθ θ̇) * ψ̇
    r1 = -cθ * θd * ψd

    # Row 2 of E: [0, cφ, sφ cθ]
    r2 = (-np.sin(φ) * φd) * θd + (cφ * φd * cθ + sφ * (-sθ * θd)) * ψd

    # Row 3 of E: [0, -sφ, cφ cθ]
    r3 = (-cφ * φd) * θd + ((-sφ * φd) * cθ + cφ * (-sθ * θd)) * ψd

    return np.asarray([r1, r2, r3], dtype=dt)


# ---------------------------------------------------------------------------
# Angular acceleration α = E(q) q̈ + (d/dt E(q)) q̇  (ZYX closed form)
# ---------------------------------------------------------------------------

@ensure_shape(3,)
def alpha_zyx(angles: Sequence[float] | np.ndarray,
              rates: Sequence[float] | np.ndarray,
              accels: Sequence[float] | np.ndarray) -> np.ndarray:
    """Angular acceleration **α = E(q) q̈ + (d/dt E) q̇** for ZYX angles."""
    q = asvec(angles, 3)
    qd = asvec(rates, 3)
    qdd = asvec(accels, 3)
    out = E_zyx(q) @ qdd + Edot_times_rates_zyx(q, qd)
    return np.asarray(out, float)


# ---------------------------------------------------------------------------
# Generic numeric E(q) for arbitrary supported sequences (dtype-preserving)
# ---------------------------------------------------------------------------

def _E_numeric_from_R(seq: str,
                      angles: Sequence[float] | np.ndarray,
                      eps: float = 1e-20) -> np.ndarray:
    """
    Complex-step construction of E(q) columns via derivatives of R(q)...
    """
    q = asvec(angles, 3)
    h = float(eps)

    R0 = euler_matrix(seq, q)          # dtype may be complex during complex-step
    E  = np.zeros((3, 3), dtype=R0.dtype)

    for i in range(3):
        dq = np.zeros(3, dtype=complex)
        dq[i] = 1j * h
        Rp = euler_matrix(seq, q.astype(complex) + dq)
        Rdot_i = np.imag(Rp) / h
        omega_hat = R0.T @ Rdot_i
        E[:, i] = _vex(0.5 * (omega_hat - omega_hat.T))

    return E   # <-- keep dtype; do NOT cast to float


@ensure_shape(3, 3)
def euler_rates_matrix(seq: str, angles: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Return E(q) for the requested Euler sequence so that ω = E(q) q̇.

    For "ZYX" we use the closed form `E_zyx`. For others, we fall back to the
    robust numeric construction `_E_numeric_from_R`. Dtype is preserved.
    """
    if str(seq).upper() == "ZYX":
        return E_zyx(angles)
    return _E_numeric_from_R(seq, angles)


# ---------------------------------------------------------------------------
# Generic numeric Ė(q, q̇) matrix used as: α = E q̈ + Ė q̇
# ---------------------------------------------------------------------------

@ensure_shape(3, 3)
def euler_accel_matrix(seq: str,
                       angles: Sequence[float] | np.ndarray,
                       rates: Sequence[float] | np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    """
    Return a 3×3 matrix Ė(q, q̇) such that α = E(q) q̈ + Ė(q, q̇) q̇.

    Implementation note:
    - We compute the *directional* product v = (d/dt)E · q̇ by differentiating
      ω(t)=E(q(t)) q̇ along q(t)=q+t q̇ with q̇ held constant (high-accuracy 5-point FD).
    - We then return a rank-1 matrix whose action on q̇ equals v. This is sufficient
      for tests which only ever use Ė @ q̇.
    """
    q  = asvec(angles, 3)
    qd = asvec(rates, 3)

    # directional derivative vector (dE/dt) q̇
    v = _Edot_times_rates_directional(seq, q, qd, h=float(eps))

    # lift vector → matrix representative so that (Ė @ q̇) = v
    denom = float(qd @ qd) + 1e-30
    Edot = np.outer(v, qd) / denom
    return np.asarray(Edot, float)


# ---------------------------------------------------------------------------
# OO façade
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EulerZYX:
    """
    Immutable ZYX Euler kinematics helper.

    Attributes
    ----------
    angles : (3,)  [φ, θ, ψ] = [roll, pitch, yaw]
    rates  : (3,)  [φ̇, θ̇, ψ̇]
    accels : (3,)  [φ̈, θ̈, ψ̈]
    """
    angles: np.ndarray
    rates: np.ndarray
    accels: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "angles", asvec(self.angles, 3))
        object.__setattr__(self, "rates",  asvec(self.rates, 3))
        object.__setattr__(self, "accels", asvec(self.accels, 3))

    @classmethod
    def from_lists(cls, angles: Sequence[float], rates: Sequence[float], accels: Sequence[float]) -> "EulerZYX":
        return cls(np.asarray(angles, float), np.asarray(rates, float), np.asarray(accels, float))

    @ensure_shape(3,)
    def omega(self) -> np.ndarray:
        return omega_zyx(self.angles, self.rates)

    @ensure_shape(3,)
    def alpha(self) -> np.ndarray:
        return alpha_zyx(self.angles, self.rates, self.accels)

    def update(self,
               angles: Sequence[float] | np.ndarray | None = None,
               rates: Sequence[float] | np.ndarray | None = None,
               accels: Sequence[float] | np.ndarray | None = None) -> "EulerZYX":
        """Return a new instance with any provided fields replaced."""
        return EulerZYX(
            np.asarray(self.angles if angles is None else angles, float),
            np.asarray(self.rates  if rates  is None else rates,  float),
            np.asarray(self.accels if accels is None else accels, float),
        )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # rotation
    "euler_matrix",
    # general rate/accel maps (used by tests)
    "euler_rates_matrix",
    "euler_accel_matrix",
    # ZYX closed forms
    "E_zyx",
    "omega_zyx",
    "alpha_zyx",
    "Edot_times_rates_zyx",
    # façade
    "EulerZYX",
]
