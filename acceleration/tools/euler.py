# acceleration/tools/euler.py
"""
Euler-angle helpers for **acceleration kinematics** (ZYX convention).

What this module provides
-------------------------
- Pure functions (NumPy-only):
    * E_zyx(angles)                     → 3×3 map s.t. ω = E(q) q̇
    * omega_zyx(angles, rates)          → 3×1 angular velocity
    * alpha_zyx(angles, rates, accels)  → 3×1 angular acceleration (closed form)
    * Edot_times_rates_zyx(angles, rates) → (d/dt E) q̇

- OO façade:
    * EulerZYX(angles, rates, accels) with:
        .omega()    → ω
        .alpha()    → α
        .update(...)  (immutable-style, returns new instance)

Why ZYX?
--------
ZYX (yaw–pitch–roll) is the most common robotics convention. If you need more
sequences later, mirror the pattern here and add E-maps with unit tests.

Shapes
------
angles = (3,)  [φ, θ, ψ]  (roll, pitch, yaw)
rates  = (3,)  [φ̇, θ̇, ψ̇]
accels = (3,)  [φ̈, θ̈, ψ̈]
ω, α   = (3,)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import numpy as np

from ..utils import ensure_shape, asvec


# ---------------------------------------------------------------------------
# Core ZYX mapping
# ---------------------------------------------------------------------------

@ensure_shape(3, 3)
def E_zyx(angles: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Return the 3×3 matrix E(q) such that **ω = E(q) q̇** for ZYX Euler angles.

    angles = [φ, θ, ψ]  (roll, pitch, yaw)

    The mapping used (common robotics convention):

        ωx =  φ̇ - ψ̇ sinθ
        ωy =  θ̇ cosφ + ψ̇ sinφ cosθ
        ωz = -θ̇ sinφ + ψ̇ cosφ cosθ

    which yields:

        E(q) = [[  1,       0,    -sinθ ],
                [  0,   cosφ,  sinφ cosθ ],
                [  0,  -sinφ,  cosφ cosθ ]]
    """
    φ, θ, ψ = asvec(angles, 3)
    sφ, cφ = np.sin(φ), np.cos(φ)
    sθ, cθ = np.sin(θ), np.cos(θ)

    E = np.array(
        [
            [1.0,     0.0,      -sθ],
            [0.0,      cφ,   sφ * cθ],
            [0.0,     -sφ,   cφ * cθ],
        ],
        dtype=float,
    )
    return E


@ensure_shape(3,)
def omega_zyx(angles: Sequence[float] | np.ndarray,
              rates: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Angular velocity **ω = E(q) q̇** for ZYX Euler angles.
    """
    q = asvec(angles, 3)
    qd = asvec(rates, 3)
    return E_zyx(q) @ qd


# ---------------------------------------------------------------------------
# Time derivative of E(q) along q̇  →  (d/dt E) q̇
# ---------------------------------------------------------------------------

@ensure_shape(3,)
def Edot_times_rates_zyx(angles: Sequence[float] | np.ndarray,
                         rates: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Compute (d/dt E(q)) q̇ for ZYX angles using the **closed-form** derivative.

    Starting from the E(q) above and differentiating wrt time:

        (d/dt E) q̇ =
            [
              -cosθ θ̇ ψ̇
              -sinφ θ̇ ψ̇ + (-sinφ φ̇) θ̇ + (cosφ)(-sinθ θ̇) ψ̇
              -cosφ θ̇ ψ̇ + (-cosφ φ̇) θ̇ + (-sinφ)(-sinθ θ̇) ψ̇
            ]

    The resulting vector corresponds to ∑_i (∂E/∂q_i q̇_i) q̇.
    This function returns a **3-vector** equal to ((d/dt)E) q̇.
    """
    φ, θ, ψ = asvec(angles, 3)
    φd, θd, ψd = asvec(rates, 3)
    sφ, cφ = np.sin(φ), np.cos(φ)
    sθ, cθ = np.sin(θ), np.cos(θ)

    # Derivative assembled directly for stability & speed.
    # Row 1 of E: [1, 0, -sθ] → time derivative ⋅ q̇ → [-cθ θ̇] * ψ̇
    r1 = -cθ * θd * ψd

    # Row 2 of E: [0, cφ, sφ cθ]
    # d/dt of [0, cφ, sφ cθ] @ q̇ = [0, (-sφ φ̇), (cφ φ̇) cθ + sφ (-sθ θ̇)] @ q̇
    # Only the third column multiplies ψ̇, second column multiplies θ̇.
    # Expanded carefully:
    r2 = (-sφ * φd) * θd + (cφ * φd * cθ + sφ * (-sθ * θd)) * ψd

    # Row 3 of E: [0, -sφ, cφ cθ]
    # d/dt of [0, -sφ, cφ cθ] @ q̇ = [0, (-cφ φ̇), (-sφ φ̇) cθ + cφ (-sθ θ̇)] @ q̇
    r3 = (-cφ * φd) * θd + ((-sφ * φd) * cθ + cφ * (-sθ * θd)) * ψd

    return np.array([r1, r2, r3], float)


# ---------------------------------------------------------------------------
# Angular acceleration α = E(q) q̈ + (d/dt E(q)) q̇
# ---------------------------------------------------------------------------

@ensure_shape(3,)
def alpha_zyx(angles: Sequence[float] | np.ndarray,
              rates: Sequence[float] | np.ndarray,
              accels: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Angular acceleration **α = E(q) q̈ + (d/dt E(q)) q̇** for ZYX angles.

    This matches the textbook relationship where E(q) is the angle-rate map for ω.
    """
    q = asvec(angles, 3)
    qd = asvec(rates, 3)
    qdd = asvec(accels, 3)
    return E_zyx(q) @ qdd + Edot_times_rates_zyx(q, qd)


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

    Methods
    -------
    omega()  → (3,) angular velocity
    alpha()  → (3,) angular acceleration
    update(angles=None, rates=None, accels=None) → EulerZYX
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
        """
        Return a new instance with any provided fields replaced.
        """
        return EulerZYX(
            np.asarray(self.angles if angles is None else angles, float),
            np.asarray(self.rates  if rates  is None else rates,  float),
            np.asarray(self.accels if accels is None else accels, float),
        )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "E_zyx",
    "omega_zyx",
    "alpha_zyx",
    "Edot_times_rates_zyx",
    "EulerZYX",
]
