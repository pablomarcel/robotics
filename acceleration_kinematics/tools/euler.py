# acceleration_kinematics/tools/euler.py
"""
Euler-angle helpers for **acceleration_kinematics kinematics**.
[...top docstring unchanged...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..utils import ensure_shape, asvec, vex as _vex


# ---------------------------------------------------------------------------
# Small SO(3) helpers
#   - Keep extended precision (longdouble) for real inputs to reduce FD noise
#   - Keep complex dtype for complex-step
#   - Cast to float64 ONLY at the end of euler_matrix when inputs are real
# ---------------------------------------------------------------------------

def _as_trig_dtype(a):
    """Use longdouble for real angles, complex for complex-step paths."""
    return np.longdouble if np.isrealobj(a) else complex

def _Rx(a: float | complex) -> np.ndarray:
    dt = _as_trig_dtype(a)
    aa = dt(a)
    ca, sa = np.cos(aa), np.sin(aa)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  ca, -sa],
                     [0.0,  sa,  ca]], dtype=type(ca))

def _Ry(a: float | complex) -> np.ndarray:
    dt = _as_trig_dtype(a)
    aa = dt(a)
    ca, sa = np.cos(aa), np.sin(aa)
    return np.array([[ ca, 0.0,  sa],
                     [0.0, 1.0, 0.0],
                     [-sa, 0.0,  ca]], dtype=type(ca))

def _Rz(a: float | complex) -> np.ndarray:
    dt = _as_trig_dtype(a)
    aa = dt(a)
    ca, sa = np.cos(aa), np.sin(aa)
    return np.array([[ ca, -sa, 0.0],
                     [ sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=type(ca))

def _R_zyx_closed_form(φ: float | complex,
                       θ: float | complex,
                       ψ: float | complex) -> np.ndarray:
    """
    Direct (intrinsic) ZYX rotation_kinematics:
        R = Rz(ψ) Ry(θ) Rx(φ)
    Built entrywise to reduce rounding vs. 3 matrix multiplies.
    """
    # choose trig dtype for each angle, then promote
    dtx = _as_trig_dtype(φ)
    dty = _as_trig_dtype(θ)
    dtz = _as_trig_dtype(ψ)
    φ = dtx(φ); θ = dty(θ); ψ = dtz(ψ)

    cφ, sφ = np.cos(φ), np.sin(φ)
    cθ, sθ = np.cos(θ), np.sin(θ)
    cψ, sψ = np.cos(ψ), np.sin(ψ)

    dt = np.result_type(cφ, sφ, cθ, sθ, cψ, sψ)
    R = np.array(
        [
            [ cψ*cθ,                 cψ*sθ*sφ - sψ*cφ,   cψ*sθ*cφ + sψ*sφ ],
            [ sψ*cθ,                 sψ*sθ*sφ + cψ*cφ,   sψ*sθ*cφ - cψ*sφ ],
            [    -sθ,                          cθ*sφ,              cθ*cφ  ],
        ],
        dtype=dt,
    )
    return R


# ---------------------------------------------------------------------------
# Directional (d/dt)E · q̇ via high-accuracy 5-point stencil (generic sequences)
#   Avoids nested complex-step; differentiates ω(q)=E(q) q̇ along q̇
# ---------------------------------------------------------------------------

def _Edot_times_rates_directional_fd(seq: str,
                                     q: np.ndarray,
                                     qd: np.ndarray,
                                     h: float = 1e-6) -> np.ndarray:
    """
    Compute v = (d/dt E(q)) q̇ along q(t) = q + t q̇ with q̇ held constant,
    using a 5-point centered stencil on ω(q) = E(q) q̇:

        v ≈ [-ω(q+2h q̇) + 8 ω(q+h q̇) - 8 ω(q-h q̇) + ω(q-2h q̇)] / (12 h)

    This is the exact vector needed in α = E q̈ + Ė q̇.
    """
    q  = asvec(q, 3)
    qd = asvec(qd, 3)
    h  = float(h)

    def omega_at(qpt: np.ndarray) -> np.ndarray:
        return np.asarray(euler_rates_matrix(seq, qpt) @ qd, float)

    ωp2 = omega_at(q + 2.0*h*qd)
    ωp1 = omega_at(q + 1.0*h*qd)
    ωm1 = omega_at(q - 1.0*h*qd)
    ωm2 = omega_at(q - 2.0*h*qd)

    return (-ωp2 + 8.0*ωp1 - 8.0*ωm1 + ωm2) / (12.0*h)


# ---------------------------------------------------------------------------
# General Euler → R (for several common sequences)
# ---------------------------------------------------------------------------

@ensure_shape(3, 3)
def euler_matrix(seq: str, angles: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Build a rotation_kinematics matrix from Euler angles for selected **intrinsic** sequences.

    Supported sequences:
      - "ZYX": R = Rz(ψ) Ry(θ) Rx(φ)      (yaw, pitch, roll)
      - "XYZ": R = Rx(α) Ry(β) Rz(γ)
      - "ZXZ": R = Rz(α) Rx(β) Rz(γ)      (proper-Euler)
      - "ZYZ": R = Rz(α) Ry(β) Rz(γ)      (proper-Euler)

    Compose in extended precision; if the result is real, return float64 so
    numpy.linalg works everywhere. For complex-step paths, keep complex.
    """
    s = str(seq).upper()
    a1, a2, a3 = asvec(angles, 3)
    if s == "ZYX":
        R = _R_zyx_closed_form(a1, a2, a3)   # roll, pitch, yaw
    elif s == "XYZ":
        R = _Rx(a1) @ _Ry(a2) @ _Rz(a3)
    elif s == "ZXZ":
        R = _Rz(a1) @ _Rx(a2) @ _Rz(a3)
    elif s == "ZYZ":
        R = _Rz(a1) @ _Ry(a2) @ _Rz(a3)
    else:
        raise ValueError(f"Unsupported Euler sequence {seq!r}; try one of 'ZYX', 'XYZ', 'ZXZ', 'ZYZ'.")
    return R.astype(float, copy=False) if np.isrealobj(R) else R


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
    sθ, cθ = np.sin(θ), np.cos(θ)
    dt = np.result_type(sφ, cφ, sθ, cθ)
    return np.array(
        [
            [1.0, 0.0,     -sθ],
            [0.0,  cφ,  sφ * cθ],
            [0.0, -sφ,  cφ * cθ],
        ],
        dtype=dt,
    )


@ensure_shape(3,)
def omega_zyx(angles: Sequence[float] | np.ndarray,
              rates: Sequence[float] | np.ndarray) -> np.ndarray:
    """Angular velocity_kinematics **ω = E(q) q̇** for ZYX angles."""
    q = asvec(angles, 3)
    qd = asvec(rates, 3)
    out = E_zyx(q) @ qd
    return np.asarray(out, float)


# ---------------------------------------------------------------------------
# Time derivative of E(q) along q̇ → (d/dt E) q̇  (ZYX closed form)
# ---------------------------------------------------------------------------

@ensure_shape(3,)
def Edot_times_rates_zyx(angles: Sequence[float] | np.ndarray,
                         rates: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Compute (d/dt E(q)) q̇ for ZYX angles using a closed-form derivative.
    """
    φ, θ, ψ = asvec(angles, 3)
    φd, θd, ψd = asvec(rates, 3)

    sφ, cφ = np.sin(φ), np.cos(φ)
    sθ, cθ = np.sin(θ), np.cos(θ)

    # Row 1 of E: [1, 0, -sθ] → time derivative ⋅ q̇ → (-cθ θ̇) * ψ̇
    r1 = -cθ * θd * ψd

    # Row 2 of E: [0, cφ, sφ cθ]
    r2 = (-np.sin(φ) * φd) * θd + (cφ * φd * cθ + sφ * (-sθ * θd)) * ψd

    # Row 3 of E: [0, -sφ, cφ cθ]
    r3 = (-cφ * φd) * θd + ((-sφ * φd) * cθ + cφ * (-sθ * θd)) * ψd

    return np.asarray([r1, r2, r3], dtype=float)


# ---------------------------------------------------------------------------
# Angular acceleration_kinematics α = E(q) q̈ + (d/dt E) q̇  (ZYX closed form)
# ---------------------------------------------------------------------------

@ensure_shape(3,)
def alpha_zyx(angles: Sequence[float] | np.ndarray,
              rates: Sequence[float] | np.ndarray,
              accels: Sequence[float] | np.ndarray) -> np.ndarray:
    """Angular acceleration_kinematics **α = E(q) q̈ + (d/dt E) q̇** for ZYX angles."""
    q = asvec(angles, 3)
    qd = asvec(rates, 3)
    qdd = asvec(accels, 3)
    out = E_zyx(q) @ qdd + Edot_times_rates_zyx(q, qd)
    return np.asarray(out, float)


# ---------------------------------------------------------------------------
# Generic numeric E(q) for arbitrary supported sequences (dtype-preserving)
#   Build E by complex-step on R:  Rᵀ ∂R/∂q_i = [E_i]^
# ---------------------------------------------------------------------------

def _E_numeric_from_R(seq: str,
                      angles: Sequence[float] | np.ndarray,
                      eps: float = 1e-20) -> np.ndarray:
    q = asvec(angles, 3)
    h = float(eps)

    R0 = euler_matrix(seq, q)
    E  = np.zeros((3, 3), dtype=R0.dtype)

    for i in range(3):
        dq = np.zeros(3, dtype=complex)
        dq[i] = 1j * h
        Rp = euler_matrix(seq, q.astype(complex) + dq)
        Rdot_i = np.imag(Rp) / h
        omega_hat = R0.T @ Rdot_i
        E[:, i] = _vex(0.5 * (omega_hat - omega_hat.T))

    return E   # keep dtype; do NOT cast to float


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
#   Build a rank-1 representative with the correct action on q̇
# ---------------------------------------------------------------------------

@ensure_shape(3, 3)
def euler_accel_matrix(seq: str,
                       angles: Sequence[float] | np.ndarray,
                       rates: Sequence[float] | np.ndarray,
                       eps: float = 1e-20) -> np.ndarray:
    """
    Return a 3×3 matrix Ė(q, q̇) such that α = E(q) q̈ + Ė(q, q̇) q̇.

    We compute the *directional* product v = (d/dt)E · q̇ using either:
      - exact closed form for ZYX (`Edot_times_rates_zyx`), or
      - a 5-point stencil on ω(q)=E(q) q̇ for other sequences
        (see `_Edot_times_rates_directional_fd`).

    We then return a rank-1 representative Ė whose action on q̇ equals v.
    """
    q  = asvec(angles, 3)
    qd = asvec(rates, 3)

    if str(seq).upper() == "ZYX":
        v = Edot_times_rates_zyx(q, qd).astype(float)
    else:
        # Avoid nested complex-step; use robust directional FD
        v = _Edot_times_rates_directional_fd(seq, q, qd, h=1e-6).astype(float)

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
    # rotation_kinematics
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
