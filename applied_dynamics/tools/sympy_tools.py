# applied_dynamics/sympy_tools.py
from __future__ import annotations
"""
Small, sharp SymPy helpers used across the **applied_dynamics** dynamics module.

Highlights
----------
- Inertia helpers:
    inertia_invariants(I), principal_axes(I), symmetrize(I), is_spd(I)

- se(3)/so(3) micro-ops (symbolic):
    skew(v), vee(S), hat3(v), vee3(S)
    blockdiag(*Ms), simplify_matrix(M)

- Generalized coordinates & Lagrange niceties:
    q_series(q_funcs, t)               → q, qd, qdd (as column matrices)
    algebraic_symbols(q_funcs)         → (q_sym, qd_sym, qdd_sym)
    replace_time_derivatives(expr, ...)→ expr(q, qd, qdd) algebraic
    total_time_derivative(expr, qf, t) → d/dt expr via chain rule
    rayleigh_quadratic(D, qd)          → R = 1/2 qdᵀ D qd, dR_dqd()

- Kinematics snippets:
    point_jacobian(pq, q)              → Jp = ∂p/∂q
    kinetic_point_mass(m, pq, q, t)    → 1/2 m ||ṗ||²

- Calculus:
    line_integral(F(x,y,z), r(t), t, t0, t1)

All functions are pure SymPy for easy testing and CI friendliness.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Callable, Optional

import sympy as sp

Matrix = sp.Matrix
Expr = sp.Expr
Symbol = sp.Symbol


# ---------------------------------------------------------------------------
# Inertia helpers
# ---------------------------------------------------------------------------

def symmetrize(I: Matrix) -> Matrix:
    """Return (I + I.T)/2 to ensure a symmetric inertia (or any 3×3 matrix)."""
    I = Matrix(I)
    return sp.simplify((I + I.T) / 2)


def inertia_invariants(I: Matrix) -> Tuple[Expr, Expr, Expr]:
    """
    Return (a1, a2, a3) invariants of a 3×3 inertia matrix (Eq. 10.214–10.216).

        a1 = tr(I)
        a2 = 1/2 (a1^2 − tr(I^2))
        a3 = det(I)
    """
    I = symmetrize(Matrix(I))
    a1 = sp.trace(I)
    a2 = sp.Rational(1, 2) * (a1**2 - sp.trace(I**2))
    a3 = I.det()
    return sp.simplify(a1), sp.simplify(a2), sp.simplify(a3)


def _orthonormalize_cols(W: Matrix) -> Matrix:
    """Gram–Schmidt orthonormalization of columns of W."""
    W = Matrix(W)
    cols: List[Matrix] = []
    for k in range(W.shape[1]):
        v = W[:, k]
        for j in range(len(cols)):
            v = v - (v.dot(cols[j])) * cols[j]
        n = sp.sqrt(sp.simplify(v.dot(v)))
        if sp.simplify(n) == 0:
            # Degenerate; keep as-is to avoid division by zero
            cols.append(v)
        else:
            cols.append(v / n)
    return Matrix.hstack(*cols) if cols else Matrix.zeros(W.rows, 0)


def principal_axes(I: Matrix) -> Tuple[Matrix, Matrix]:
    """
    Compute principal moments and axes for a (symmetric) inertia matrix.

    Returns
    -------
    (vals, W) :
        vals : 3×1 column of eigenvalues (sorted ascending)
        W    : 3×3 orthonormal eigenvectors (columns = axes)

    Notes
    -----
    - Symmetrizes I for robustness.
    - If multiplicities>1, eigenvects can return multiple vectors; we stack and
      Gram–Schmidt to orthonormalize. Sorting is by numeric value of eigenvalue.
    """
    I = symmetrize(Matrix(I))
    evects = I.eigenvects()  # [(eval, mult, [vecs]), ...]
    vals: List[Expr] = []
    vecs: List[Matrix] = []
    for lam, _mult, vlist in evects:
        for v in vlist:
            vv = Matrix(v)
            nn = sp.sqrt(sp.simplify(vv.dot(vv)))
            vecs.append(vv if sp.simplify(nn) == 0 else vv / nn)
            vals.append(sp.simplify(lam))
    # Fallback if eigenvects gives only one vector for repeated roots
    if len(vecs) < 3:
        # Try to complete a basis numerically-then-symbolically
        from itertools import product
        basis = [Matrix([1,0,0]), Matrix([0,1,0]), Matrix([0,0,1])]
        for b in basis:
            if len(vecs) == 3: break
            if all(sp.simplify(b.dot(v)) != 1 for v in vecs):
                vecs.append(b)
                vals.append(vals[-1] if vals else sp.trace(I)/3)
    pairs = sorted(zip(vals, vecs), key=lambda t: sp.N(t[0]))
    vals_s, vecs_s = zip(*pairs)
    W = _orthonormalize_cols(Matrix.hstack(*vecs_s))
    return Matrix(vals_s).reshape(3, 1), W


def is_spd(I: Matrix) -> bool:
    """
    Quick check (symbolic-friendly) for symmetric positive-definite:
      - symmetric
      - all principal minors positive (Sylvester’s criterion)
    """
    I = symmetrize(Matrix(I))
    try:
        m1 = I[0, 0]
        m2 = I[:2, :2].det()
        m3 = I.det()
        return all(sp.simplify(mi) > 0 for mi in (m1, m2, m3))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# so(3)/se(3) micro-ops (symbolic)
# ---------------------------------------------------------------------------

def skew(v: Matrix) -> Matrix:
    """Skew-symmetric matrix [v]^ for v in R^3 (so that [v]^ w = v × w)."""
    v = Matrix(v).reshape(3, 1)
    x, y, z = v
    return Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def vee(S: Matrix) -> Matrix:
    """Inverse of skew: vee([v]^) = v (expects a 3×3 skew-symmetric)."""
    S = Matrix(S)
    return Matrix([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2


def hat3(v: Matrix) -> Matrix:
    """Alias for skew for clarity in some derivations."""
    return skew(v)


def vee3(S: Matrix) -> Matrix:
    """Alias for vee for clarity."""
    return vee(S)


def blockdiag(*Ms: Matrix) -> Matrix:
    """Block-diagonal concatenation."""
    if not Ms:
        return Matrix([])
    out = Ms[0]
    for M in Ms[1:]:
        out = sp.diag(out, M)
    return Matrix(out)


def simplify_matrix(M: Matrix) -> Matrix:
    """Elementwise simplify; keeps matrix shape."""
    M = Matrix(M)
    return M.applyfunc(sp.simplify)


# ---------------------------------------------------------------------------
# Generalized coordinates helpers
# ---------------------------------------------------------------------------

def q_series(q_funcs: Sequence[sp.Function], t: Symbol) -> Tuple[Matrix, Matrix, Matrix]:
    """
    Build q(t), q̇(t), q̈(t) from a list of sympy.Function constructors.
    """
    q = Matrix([f(t) for f in q_funcs])
    qd = Matrix([sp.diff(f(t), t) for f in q_funcs])
    qdd = Matrix([sp.diff(f(t), (t, 2)) for f in q_funcs])
    return q, qd, qdd


def algebraic_symbols(q_funcs: Sequence[sp.Function]) -> Tuple[List[Symbol], List[Symbol], List[Symbol]]:
    """
    Return algebraic (time-free) symbols (q, qd, qdd) corresponding to q_i(t).
    """
    def base(f): return getattr(f, "__name__", str(f))
    q_sym   = [sp.Symbol(f"{base(f)}") for f in q_funcs]
    qd_sym  = [sp.Symbol(f"{base(f)}d") for f in q_funcs]
    qdd_sym = [sp.Symbol(f"{base(f)}dd") for f in q_funcs]
    return q_sym, qd_sym, qdd_sym


def replace_time_derivatives(expr: Expr,
                             q_funcs: Sequence[sp.Function],
                             t: Symbol,
                             q_sym: Sequence[Symbol],
                             qd_sym: Sequence[Symbol],
                             qdd_sym: Sequence[Symbol]) -> Expr:
    """
    Replace q_i(t), q̇_i(t), q̈_i(t) with algebraic symbols for coefficient extraction.
    """
    repl: Dict[Expr, Expr] = {}
    for f, qs, qds, qdds in zip(q_funcs, q_sym, qd_sym, qdd_sym):
        repl[f(t)] = qs
        repl[sp.diff(f(t), t)] = qds
        repl[sp.diff(f(t), (t, 2))] = qdds
    return sp.simplify(sp.expand(expr.xreplace(repl)))


def total_time_derivative(expr: Expr, q_funcs: Sequence[sp.Function], t: Symbol) -> Expr:
    """
    Total derivative d/dt of expr(q(t), q̇(t), t) via chain rule.
    Useful when expr depends explicitly on t, q, and qd.

    d/dt expr = ∂expr/∂t + Σ (∂expr/∂q_i) q̇_i + Σ (∂expr/∂q̇_i) q̈_i
    """
    q, qd, qdd = q_series(q_funcs, t)
    d_expr = sp.diff(expr, t)
    for i in range(len(q_funcs)):
        d_expr += sp.diff(expr, q[i]) * qd[i] + sp.diff(expr, qd[i]) * qdd[i]
    return sp.simplify(d_expr)


def rayleigh_quadratic(D: Matrix, qd: Matrix) -> Tuple[Expr, Matrix]:
    """
    Build Rayleigh dissipation R = 1/2 q̇ᵀ D q̇ and its gradient dR/dq̇.
    """
    D = Matrix(D)
    qd = Matrix(qd)
    R = sp.Rational(1, 2) * (qd.T * D * qd)[0]
    dR_dqd = sp.Matrix([sp.diff(R, qi) for qi in qd])
    return sp.simplify(R), simplify_matrix(dR_dqd)


# ---------------------------------------------------------------------------
# Kinematics snippets
# ---------------------------------------------------------------------------

def point_jacobian(pq: Matrix, q: Matrix) -> Matrix:
    """
    Jacobian of a point position wrt generalized coordinates:
        Jp = ∂p/∂q (p is 3×1, q is n×1)
    """
    pq = Matrix(pq).reshape(3, 1)
    q = Matrix(q).reshape(-1, 1)
    J = pq.jacobian(q)
    return Matrix(J)


def kinetic_point_mass(m: Expr, p_q: Matrix, q_funcs: Sequence[sp.Function], t: Symbol) -> Expr:
    """
    Kinetic energy for a point mass following p(q(t)):
        K = 1/2 m ||ṗ||²
    """
    p_q = Matrix(p_q).reshape(3, 1)
    q, qd, _ = q_series(q_funcs, t)
    # dp/dt = (∂p/∂q) q̇
    Jp = p_q.jacobian(q)
    pd = Jp * qd
    return sp.simplify(sp.Rational(1, 2) * m * (pd.dot(pd)))


# ---------------------------------------------------------------------------
# Calculus: line integral
# ---------------------------------------------------------------------------

def line_integral(F_xyz: Sequence[Expr], r_of_t: Matrix, t_symbol: Symbol, t0: Expr, t1: Expr) -> Expr:
    """
    Compute ∫ F·dr over t ∈ [t0, t1], where F is given as [Fx(x,y,z), Fy(x,y,z), Fz(x,y,z)]
    and r(t) is a 3×1 curve.

    We substitute x,y,z → r(t) safely and integrate F(r(t)) · ṙ(t) dt.
    """
    r_of_t = Matrix(r_of_t).reshape(3, 1)
    Fx, Fy, Fz = list(F_xyz)
    x, y, z = sp.symbols("x y z")
    subs_map = {x: r_of_t[0], y: r_of_t[1], z: r_of_t[2]}
    F_sub = Matrix([Fx.subs(subs_map), Fy.subs(subs_map), Fz.subs(subs_map)])
    drdt = Matrix([sp.diff(r_of_t[i], t_symbol) for i in range(3)])
    integrand = sp.simplify(F_sub.dot(drdt))
    return sp.integrate(integrand, (t_symbol, t0, t1))
