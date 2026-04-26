# applied_dynamics/core.py
"""Core object model for **applied_dynamics dynamics** (Lagrangian mechanics).

What’s inside
-------------
- Inertia                 — rigid-body mass properties & parallel-axis shift
- FrameState              — (q, q̇) container used by energy providers
- Energy (Protocol)       — kinetic() / potential() contract
- System (ABC)            — base class for Lagrange-ready systems:
    * lagrangian_state()           → (q, qd, t)
    * energy()                     → Energy
    * generalized_forces(fs)       → Q(q, q̇, t)
    * lagrangian()                 → K - V
    * mass_matrix()                → ∂²K/∂q̇∂q̇ᵀ
    * equations_of_motion()        → Euler–Lagrange LHS

Design notes
------------
- All symbolic work uses SymPy (matrices & expressions).
- We **lazy-import** :mod:`applied_dynamics.dynamics` inside helpers to avoid cycles.
- Models in :mod:`applied_dynamics.models` implement :class:`System` and can reuse
  these convenience methods directly.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Protocol, Tuple, List

import sympy as sp

# Re-export common SymPy types for clarity
Matrix = sp.Matrix
Expr = sp.Expr
Symbol = sp.Symbol


# ---------------------------------------------------------------------------
# Small helper
# ---------------------------------------------------------------------------

def as_column(x) -> Matrix:
    """Return `x` as a SymPy **column** Matrix."""
    m = sp.Matrix(x)
    return m if m.shape[1] == 1 else m.reshape(m.size, 1)


# ---------------------------------------------------------------------------
# Rigid body mass properties
# ---------------------------------------------------------------------------

Vec3 = sp.Matrix  # 3×1 vector


@dataclass
class Inertia:
    """Rigid-body mass properties expressed in the **body** frame.

    Parameters
    ----------
    m : Symbol | float
        Mass.
    I : Matrix (3×3)
        Inertia about the body-frame origin (typically CoM frame).
    com_B : Vec3, optional
        Center of mass coordinates in the body frame, default 0.

    Notes
    -----
    Parallel-axis shift uses
        I_shifted = I + m (||r||² I₃ - r rᵀ)
    """
    m: Symbol | float
    I: Matrix
    com_B: Vec3 = field(default_factory=lambda: sp.Matrix([0, 0, 0]))

    def parallel_axis(self, r_BC: Vec3) -> "Inertia":
        """Return inertia shifted by vector ``r_BC`` (body frame)."""
        r = sp.Matrix(r_BC)
        I_shift = self.I + self.m * (r.dot(r) * sp.eye(3) - r * r.T)
        return Inertia(self.m, sp.simplify(I_shift), self.com_B + r)


# ---------------------------------------------------------------------------
# State & energy contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameState:
    """Pose-independent generalized state for Lagrangian systems."""
    q: Matrix     # generalized coordinates (column)
    qd: Matrix    # generalized velocities (column)


class Energy(Protocol):
    """Kinetic & potential energy providers for a system."""
    def kinetic(self, fs: FrameState) -> Expr: ...
    def potential(self, fs: FrameState) -> Expr: ...


# ---------------------------------------------------------------------------
# Base system for Lagrange workflows
# ---------------------------------------------------------------------------

class System(abc.ABC):
    """
    Minimal abstract base class for systems amenable to Lagrange's equations.

    Subclasses must implement:
      - energy(self) -> Energy
      - generalized_forces(self, fs) -> Matrix
      - lagrangian_state(self) -> (q, qd, t)

    The helpers `lagrangian()`, `mass_matrix()`, and `equations_of_motion()`
    are provided here and use :mod:`applied_dynamics.dynamics` (lazy-imported).
    """

    # ----- abstract API the models must provide -----

    @abc.abstractmethod
    def energy(self) -> Energy:
        ...

    @abc.abstractmethod
    def generalized_forces(self, fs: FrameState) -> Matrix:
        ...

    @abc.abstractmethod
    def lagrangian_state(self) -> Tuple[Matrix, Matrix, Symbol]:
        """
        Returns
        -------
        q : Matrix
            Column of generalized coordinates, e.g. Matrix([θ(t), φ(t), ...]).
        qd : Matrix
            Column of generalized velocities, e.g. Matrix([θ̇(t), φ̇(t), ...]).
        t : Symbol
            Time symbol used by the system (typically ``sp.symbols('t')``).
        """
        ...

    # ----- convenience helpers used in models/tests -----

    def lagrangian(self) -> Expr:
        """Compute ``L = K - V`` from the model's :class:`Energy` (symbolic)."""
        q, qd, _ = self.lagrangian_state()
        fs = FrameState(as_column(q), as_column(qd))
        K = self.energy().kinetic(fs)
        V = self.energy().potential(fs)
        return sp.simplify(K - V)

    def mass_matrix(self) -> Matrix:
        """Return the mass matrix ``M(q) = ∂²K/∂q̇∂q̇ᵀ``."""
        from .dynamics import mass_matrix_from_lagrangian  # lazy import
        q, qd, _ = self.lagrangian_state()
        fs = FrameState(as_column(q), as_column(qd))
        K = sp.simplify(self.energy().kinetic(fs))
        return mass_matrix_from_lagrangian(K, list(fs.qd))

    def equations_of_motion(self) -> Matrix:
        """
        Return the Euler–Lagrange equations (left-hand side), i.e.,
            d/dt(∂L/∂q̇) − ∂L/∂q − Q = 0
        so the resulting vector equals 0 at dynamics satisfaction.

        Uses :class:`applied_dynamics.dynamics.LagrangeEngine`.
        """
        from .dynamics import LagrangeEngine  # lazy import to avoid cycles

        q_sym, qd_sym, t = self.lagrangian_state()
        q_sym = as_column(q_sym)
        qd_sym = as_column(qd_sym)

        # Recover **function constructors** from entries like "theta(t)" → Function('theta')
        q_funcs: List[sp.Function] = []
        for idx, expr in enumerate(q_sym, start=1):
            s = str(expr)
            name = s.split("(", 1)[0] if "(" in s else f"q{idx}"
            q_funcs.append(sp.Function(name))

        # Energies with the same symbolic state
        fs = FrameState(q_sym, qd_sym)
        K = sp.simplify(self.energy().kinetic(fs))
        V = sp.simplify(self.energy().potential(fs))

        # Generalized forces → ensure a proper (n,1) column; default zeros if None.
        Q_raw = self.generalized_forces(fs)
        if Q_raw is None:
            Q = sp.Matrix.zeros(q_sym.shape[0], 1)
        else:
            Q = sp.Matrix(Q_raw)
            if Q.shape == (q_sym.shape[0],):
                Q = Q.reshape(q_sym.shape[0], 1)
            assert Q.shape == (q_sym.shape[0], 1), "generalized_forces must be a column vector of size dof"

        engine = LagrangeEngine()
        return engine.equations_of_motion(q_funcs, t, K, V, Q)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "Inertia",
    "FrameState",
    "Energy",
    "System",
    "Matrix",
    "Expr",
    "Symbol",
    "as_column",
]
