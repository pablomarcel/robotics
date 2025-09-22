# applied/dynamics.py
"""
Symbolic dynamics helpers for the **applied** module.

What’s inside
-------------
- LagrangeEngine
    * equations_of_motion(q_funcs, t, K, V, Q=None, R=None)
        → Euler–Lagrange with optional Rayleigh dissipation.
    * linearize_mqb(eoms, q_funcs, t)
        → factor EOMs as M(q) qdd + b(q, qd) = 0 (LHS form).

- mass_matrix_from_lagrangian(K, qd)
    → symmetric Hessian wrt q̇

- NewtonEuler
    → small helpers for momenta & cross ops (optional use)

Design goals
------------
- Pure SymPy (no external dependencies).
- No circular imports to models or app code.
- Keep equations **left-hand side** = 0 for testability and composition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict

import sympy as sp


# ---------------------------------------------------------------------------
# Utilities: symbolic packs & total derivatives
# ---------------------------------------------------------------------------

def _ensure_functions(q_funcs: Sequence[sp.Function]) -> Tuple[List[sp.Function],]:
    """Validate q_funcs is a sequence of sympy.Function (constructors)."""
    QF: List[sp.Function] = []
    for f in q_funcs:
        if not isinstance(f, sp.FunctionClass) and not isinstance(f, sp.Function):
            # Allow e.g. expr.func which is a FunctionClass; otherwise raise.
            raise TypeError("q_funcs must be a sequence of sympy function constructors (e.g., Function('theta')).")
        # Normalize to FunctionClass
        QF.append(f if isinstance(f, sp.FunctionClass) else f.func)  # type: ignore
    return (QF,)


def _q_series(q_funcs: Sequence[sp.Function], t: sp.Symbol) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Build column vectors:
        q(t)   = [ q1(t), ..., qn(t) ]^T
        qd(t)  = [  d/dt q1(t), ..., ]^T
        qdd(t) = [ d2/dt2 q1(t), ..., ]^T
    """
    q_list  = [f(t) for f in q_funcs]
    qd_list = [sp.diff(f(t), t) for f in q_funcs]
    qdd_list = [sp.diff(f(t), (t, 2)) for f in q_funcs]
    return sp.Matrix(q_list), sp.Matrix(qd_list), sp.Matrix(qdd_list)


def _algebraic_symbols(q_funcs: Sequence[sp.Function]) -> Tuple[List[sp.Symbol], List[sp.Symbol], List[sp.Symbol]]:
    """
    Create algebraic symbols (no explicit time) for collecting coefficients:
        q_sym, qd_sym, qdd_sym
    """
    def base_name(f) -> str:
        return getattr(f, "__name__", str(f))
    q_sym   = [sp.Symbol(f"{base_name(f)}") for f in q_funcs]
    qd_sym  = [sp.Symbol(f"{base_name(f)}d") for f in q_funcs]
    qdd_sym = [sp.Symbol(f"{base_name(f)}dd") for f in q_funcs]
    return q_sym, qd_sym, qdd_sym


def _replace_time_derivatives(expr: sp.Expr,
                              q_funcs: Sequence[sp.Function],
                              t: sp.Symbol,
                              q_sym: Sequence[sp.Symbol],
                              qd_sym: Sequence[sp.Symbol],
                              qdd_sym: Sequence[sp.Symbol]) -> sp.Expr:
    """
    Replace q_i(t), q̇_i(t), q̈_i(t) with algebraic symbols for coefficient extraction.
    """
    repl: Dict[sp.Expr, sp.Expr] = {}
    for f, qs, qds, qdds in zip(q_funcs, q_sym, qd_sym, qdd_sym):
        qt   = f(t)
        qdt  = sp.diff(f(t), t)
        qddt = sp.diff(f(t), (t, 2))
        repl[qt]   = qs
        repl[qdt]  = qds
        repl[qddt] = qdds
    return sp.simplify(sp.expand(expr.xreplace(repl)))


# ---------------------------------------------------------------------------
# Core: Lagrange Engine
# ---------------------------------------------------------------------------

@dataclass
class LagrangeEngine:
    """
    Generic Euler–Lagrange equations generator.

    The returned EOM vector is the **left-hand side**:
        E(q, q̇, q̈, t) = d/dt(∂L/∂q̇) − ∂L/∂q + ∂R/∂q̇ − Q  =  0
    where R(q̇) is the Rayleigh dissipation (optional) and Q are generalized
    non-conservative forces (column vector).
    """
    use_rayleigh: bool = True

    def equations_of_motion(
        self,
        q_funcs: Sequence[sp.Function],
        t: sp.Symbol,
        K: sp.Expr,
        V: sp.Expr,
        Q: sp.Matrix | None = None,
        R: sp.Expr | None = None,
    ) -> sp.Matrix:
        """
        Compute Euler–Lagrange equations for a set of generalized coordinates.

        Parameters
        ----------
        q_funcs : sequence of sympy.Function
            Constructors for q_i (e.g., Function('theta')).
        t : Symbol
            Time symbol.
        K, V : Expr
            Kinetic and potential energy.
        Q : Matrix, optional
            Generalized non-conservative forces column vector.
        R : Expr, optional
            Rayleigh dissipation function R(q̇). If provided (and use_rayleigh),
            contributes +∂R/∂q̇ on the LHS (so physical damping is -∂R/∂q̇ on the RHS).

        Returns
        -------
        Matrix
            Column vector of EOM **left-hand sides** (equals zero at satisfaction).
        """
        (q_funcs,) = _ensure_functions(q_funcs)
        n = len(q_funcs)
        Qcol = sp.Matrix.zeros(n, 1) if Q is None else sp.Matrix(Q).reshape(n, 1)

        L = sp.simplify(K - V)
        eoms: List[sp.Expr] = []

        for i, qi in enumerate(q_funcs):
            q_i  = qi(t)
            qdi  = sp.diff(qi(t), t)

            dL_dq  = sp.diff(L, q_i)
            dL_dqd = sp.diff(L, qdi)
            ddt_dL_dqd = sp.diff(dL_dqd, t)

            lhs = sp.simplify(ddt_dL_dqd - dL_dq)

            # Optional Rayleigh dissipation (adds +∂R/∂q̇ to LHS)
            if self.use_rayleigh and (R is not None):
                lhs += sp.diff(R, qdi)

            # Subtract generalized forces to keep it all on the **left**
            lhs -= Qcol[i, 0]

            eoms.append(sp.simplify(sp.expand(lhs)))

        return sp.Matrix(eoms)

    # -------------------- Factorization helpers --------------------

    def linearize_mqb(
        self,
        eoms: sp.Matrix,
        q_funcs: Sequence[sp.Function],
        t: sp.Symbol,
    ) -> Tuple[sp.Matrix, sp.Matrix, List[sp.Symbol]]:
        """
        Factor **left-hand side** equations as:
            M(q) q̈ + b(q, q̇) = 0

        This is done by replacing q(t), q̇(t), q̈(t) with algebraic symbols,
        extracting coefficients of q̈, and putting the remainder in b.

        Returns
        -------
        (M, b, qdd_symbols)
            - M : (n×n) mass matrix (symbolic)
            - b : (n×1) symbolic remainder so that M*qdd + b = 0
            - qdd_symbols : the algebraic symbols used for q̈ (for later substitution)
        """
        (q_funcs,) = _ensure_functions(q_funcs)
        n = len(q_funcs)
        q_sym, qd_sym, qdd_sym = _algebraic_symbols(q_funcs)

        M = sp.zeros(n, n)
        b = sp.zeros(n, 1)

        for i in range(n):
            # Replace time functions with algebraic symbols.
            ei = _replace_time_derivatives(eoms[i, 0], q_funcs, t, q_sym, qd_sym, qdd_sym)

            # Build the i-th row of M from coefficients wrt qdd symbols.
            row_coeffs = [sp.simplify(sp.diff(ei, qdd_sym[j])) for j in range(n)]
            for j, cij in enumerate(row_coeffs):
                M[i, j] = cij

            # b_i is the leftover after removing the qdd terms.
            ei_no_qdd = sp.simplify(ei - sum(M[i, j] * qdd_sym[j] for j in range(n)))
            b[i, 0] = sp.simplify(ei_no_qdd)

        return M, b, qdd_sym


# ---------------------------------------------------------------------------
# Classic robotics extraction
# ---------------------------------------------------------------------------

def mass_matrix_from_lagrangian(K: sp.Expr, qd: Sequence[sp.Expr]) -> sp.Matrix:
    """
    Extract the mass matrix M(q) from a quadratic kinetic energy:
        K(q, q̇) ≈ 1/2 q̇ᵀ M(q) q̇
    via Hessian wrt q̇, symmetrized for robustness.
    """
    qd_vec = sp.Matrix(qd)
    H = sp.hessian(sp.expand(K), qd_vec)
    return sp.simplify((H + H.T) / 2)


# ---------------------------------------------------------------------------
# Newton–Euler mini-helpers (optional)
# ---------------------------------------------------------------------------

@dataclass
class NewtonEuler:
    """Small helpers for momentum & cross-ops."""

    @staticmethod
    def cross_mat(v: sp.Matrix) -> sp.Matrix:
        """Skew-symmetric cross product matrix [v]^ such that [v]^ w = v × w."""
        v = sp.Matrix(v).reshape(3, 1)
        x, y, z = v[0], v[1], v[2]
        return sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def linear_momentum(self, m: sp.Symbol | float, v: sp.Matrix) -> sp.Matrix:
        """p = m v"""
        return sp.Matrix(v) * sp.Matrix([m])[0]

    def angular_momentum_com(self, I_C: sp.Matrix, omega: sp.Matrix) -> sp.Matrix:
        """H_C = I_C ω"""
        return sp.Matrix(I_C) * sp.Matrix(omega)

    def shift_angular_momentum(self, H_C: sp.Matrix, r_PC: sp.Matrix, p: sp.Matrix) -> sp.Matrix:
        """Shift angular momentum from CoM to point P: H_P = H_C + r_PC × p."""
        return sp.Matrix(H_C) + sp.Matrix(r_PC).cross(sp.Matrix(p))

    def resultant_force(self, p: sp.Matrix, t: sp.Symbol) -> sp.Matrix:
        """F = d/dt p"""
        return sp.diff(sp.Matrix(p), t)

    def resultant_moment_about(self, H_P: sp.Matrix, t: sp.Symbol) -> sp.Matrix:
        """M_P = d/dt H_P"""
        return sp.diff(sp.Matrix(H_P), t)
