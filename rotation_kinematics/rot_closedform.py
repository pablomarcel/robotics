# =============================
# File: rotation_kinematics/rot_closedform.py
# =============================
from __future__ import annotations
from typing import Tuple

import sympy as sp

# -----------------------------------------------------------------------------
# Elementary active rotations (symbolic)
# -----------------------------------------------------------------------------
def _Rx(a: sp.Symbol) -> sp.Matrix:
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[1, 0, 0],
                      [0, c, -s],
                      [0, s,  c]])

def _Ry(a: sp.Symbol) -> sp.Matrix:
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]])

def _Rz(a: sp.Symbol) -> sp.Matrix:
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])

_AX = {'x': _Rx, 'y': _Ry, 'z': _Rz}

# -----------------------------------------------------------------------------
# Composition helpers
# -----------------------------------------------------------------------------
def _compose_local(seq: str, q: Tuple[sp.Symbol, sp.Symbol, sp.Symbol]) -> sp.Matrix:
    """
    Local/intrinsic composition (moving axes), applied_dynamics in the LISTED order by
    RIGHT-multiplication (starting from identity):
        R = R_axis1(q1) * R_axis2(q2) * R_axis3(q3)
    This matches SciPy's intrinsic (uppercase) convention.
    """
    s = seq.lower()
    if len(s) != 3:
        raise ValueError("sequence must have length 3")
    R1 = _AX[s[0]](q[0])
    R2 = _AX[s[1]](q[1])
    R3 = _AX[s[2]](q[2])
    return R1 * R2 * R3

def _vee(S: sp.Matrix) -> sp.Matrix:
    """vee: skew(3) -> R^3 (symbolic)."""
    return sp.Matrix([S[2, 1], S[0, 2], S[1, 0]])

# -----------------------------------------------------------------------------
# Closed-form E(q) such that ω = E(q) q̇
# -----------------------------------------------------------------------------
def E_matrix(
    seq: str,
    convention: str = 'local',
    frame: str = 'body',
    simplify: bool = True,
) -> Tuple[sp.Matrix, Tuple[sp.Symbol, sp.Symbol, sp.Symbol]]:
    """
    Return symbolic E(q) such that ω = E(q) q̇ for a 3-angle sequence.

    - convention: 'local' (intrinsic) uses RIGHT-multiplication: R = R_a1(q1) R_a2(q2) R_a3(q3)
      'global' (extrinsic) is mapped via the standard identity:
         global(seq, a1,a2,a3) == local(seq[::-1], a3,a2,a1)
    - frame: 'body' uses Ω_b = Rᵀ Ṙ, 'space' uses Ω_s = Ṙ Rᵀ
    - columns of E are in the SAME order as (a1,a2,a3) for the input seq.
    """
    seq = seq.lower()
    if convention not in {'local', 'global'}:
        raise ValueError("convention must be 'local' or 'global'")
    if frame not in {'body', 'space'}:
        raise ValueError("frame must be 'body' or 'space'")

    a1, a2, a3 = sp.symbols('a1 a2 a3', real=True)
    q = (a1, a2, a3)

    if convention == 'local':
        R = _compose_local(seq, q)
    else:
        # global(seq, a1,a2,a3) == local(seq[::-1], a3,a2,a1)
        R = _compose_local(seq[::-1], (a3, a2, a1))

    Rt = R.T
    cols = []
    for ak in q:
        dR = sp.diff(R, ak)
        Om = dR * Rt if frame == 'space' else Rt * dR
        Om = (Om - Om.T) / 2  # enforce skew
        cols.append(_vee(Om))

    E = sp.Matrix.hstack(*cols)
    if simplify:
        E = sp.simplify(E)
    return E, (a1, a2, a3)

# -----------------------------------------------------------------------------
# RPY (ZYX) convenience: sequence order vs. classic RPY order
# -----------------------------------------------------------------------------
def E_matrix_rpy_zyx_body(
    convention: str = 'local',
    simplify: bool = True,
) -> Tuple[sp.Matrix, Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
           sp.Matrix, Tuple[sp.Symbol, sp.Symbol, sp.Symbol]]:
    """
    Return BOTH:
      1) E in *sequence* order [a1=ψ (z), a2=θ (y), a3=φ (x)]
      2) E reindexed to classic RPY order [φ, θ, ψ] (columns aligned to [φ̇, θ̇, ψ̇])
    """
    E_seq, (a1, a2, a3) = E_matrix('zyx', convention=convention, frame='body', simplify=simplify)
    ψ, θ, φ = a1, a2, a3
    # reorder columns from [ψ̇,θ̇,φ̇] → [φ̇,θ̇,ψ̇]
    E_rpy = E_seq[:, [2, 1, 0]]
    if simplify:
        E_rpy = sp.simplify(E_rpy)
    return E_seq, (a1, a2, a3), E_rpy, (φ, θ, ψ)

def E_matrix_proper(
    seq: str,
    convention: str = 'local',
    frame: str = 'body',
    simplify: bool = True,
) -> Tuple[sp.Matrix, Tuple[sp.Symbol, sp.Symbol, sp.Symbol]]:
    """Convenience wrapper for proper Euler sequences (zxz, zyz, xyx, xzx, yxy, yzy)."""
    s = seq.lower()
    if s not in {'zxz', 'zyz', 'xyx', 'xzx', 'yxy', 'yzy'}:
        raise ValueError("E_matrix_proper expects a proper Euler sequence like 'zxz' or 'zyz'")
    return E_matrix(s, convention=convention, frame=frame, simplify=simplify)
