# robot/dynamics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np
import sympy as sp

from .core import RobotModel, State


# ======================================================================
# Public protocol and output container
# ======================================================================

class DynamicsEngine(Protocol):
    def compute(self, model: RobotModel, state: State, gravity: float) -> "DynOut": ...


@dataclass(slots=True)
class DynOut:
    """Numerical dynamics result."""
    M: np.ndarray   # (n,n) inertia matrix
    C: np.ndarray   # (n,n) Coriolis/centrifugal matrix (so that C@qd == H)
    g: np.ndarray   # (n,) gravity vector
    tau: np.ndarray # (n,) tau = M*qdd + C@qd + g


# ======================================================================
# SymPy Lagrange Engine (planar 2R reference implementation)
# ======================================================================

class SympyLagrangeEngine:
    """
    Symbolic Lagrange engine for a planar 2R manipulator.

    Assumptions
    -----------
    - Planar motion in XY with gravity along -Y (i.e., +y upward for potential).
    - Link i has mass m_i, length l_i, COM at c_i * l_i from its proximal joint along x_i.
    - Each link is a slender rod in plane: Izz_i = m_i*l_i^2/12.
    - Returns C such that (C @ qd) equals the usual H(q,qd) vector exactly.

    Notes
    -----
    This engine is intentionally small and fast for tests. For larger models,
    add more engines (e.g., Pinocchio) that implement `DynamicsEngine`.
    """

    def __init__(self) -> None:
        # Generalized coordinates/speeds/accelerations
        q1, q2 = sp.symbols("q1 q2", real=True)
        qd1, qd2 = sp.symbols("qd1 qd2", real=True)
        qdd1, qdd2 = sp.symbols("qdd1 qdd2", real=True)

        # Parameters
        l1, l2, m1, m2, c1, c2, g = sp.symbols("l1 l2 m1 m2 c1 c2 g", positive=True, real=True)

        q = sp.Matrix([q1, q2])
        qd = sp.Matrix([qd1, qd2])
        qdd = sp.Matrix([qdd1, qdd2])

        # ------------------------------------------------------------------
        # Kinematics: COM positions in base frame
        # ------------------------------------------------------------------
        # C1 at (c1*l1) from joint 1 along link 1
        x1 = c1*l1*sp.cos(q1)
        y1 = c1*l1*sp.sin(q1)
        # C2 at end of link1 + (c2*l2) along link2
        x2 = l1*sp.cos(q1) + c2*l2*sp.cos(q1 + q2)
        y2 = l1*sp.sin(q1) + c2*l2*sp.sin(q1 + q2)

        p1 = sp.Matrix([x1, y1])
        p2 = sp.Matrix([x2, y2])

        # Jacobians for translational velocities
        Jv1 = p1.jacobian(q)  # 2x2
        Jv2 = p2.jacobian(q)  # 2x2
        v1 = Jv1 * qd
        v2 = Jv2 * qd

        # ------------------------------------------------------------------
        # Kinetic and potential energy
        # ------------------------------------------------------------------
        Izz1 = m1 * l1**2 / 12
        Izz2 = m2 * l2**2 / 12

        K_trans = sp.Rational(1, 2) * (m1 * (v1.dot(v1)) + m2 * (v2.dot(v2)))
        K_rot = sp.Rational(1, 2) * (Izz1 * qd1**2 + Izz2 * (qd1 + qd2)**2)
        K = sp.simplify(K_trans + K_rot)

        # Potential: y is positive upward; gravity points -y
        V = m1 * g * y1 + m2 * g * y2

        # Inertia matrix M = ∂^2 K / ∂qd∂qd
        M = sp.hessian(K, qd)

        # Gravity vector G(q) = ∂V / ∂q
        G = sp.Matrix([sp.diff(V, q1), sp.diff(V, q2)])

        # ------------------------------------------------------------------
        # Coriolis/Centrifugal matrix C built so that C(q,qd)@qd == H(q,qd)
        # using Christoffel symbols of the first kind:
        #   c_ijk = 1/2 ( ∂M_ij/∂q_k + ∂M_ik/∂q_j - ∂M_jk/∂q_i )
        # Then (C @ qd)_i = Σ_j Σ_k c_ijk * qd_j * qd_k
        # We collect terms linear in qd to return a matrix C(q,qd).
        # ------------------------------------------------------------------
        dM_dq1 = M.applyfunc(lambda e: sp.diff(e, q1))
        dM_dq2 = M.applyfunc(lambda e: sp.diff(e, q2))
        dM = [dM_dq1, dM_dq2]  # index by k: 0->q1, 1->q2

        C = sp.Matrix.zeros(2, 2)  # so that C @ qd == H
        for i in range(2):
            for j in range(2):
                expr = 0
                for k in range(2):
                    c_ijk = sp.Rational(1, 2) * (dM[k][i, j] + dM[j][i, k] - dM[i][j, k])
                    expr += c_ijk * qd[k]
                C[i, j] = sp.simplify(expr)

        # τ = M q̈ + C q̇ + G
        tau = sp.simplify(M * qdd + C * qd + G)

        # Lambdify for fast numeric calls
        args = (q1, q2, qd1, qd2, qdd1, qdd2, l1, l2, m1, m2, c1, c2, g)
        self._fM = sp.lambdify(args, M, "numpy")
        self._fC = sp.lambdify(args, C, "numpy")
        self._fG = sp.lambdify(args, G, "numpy")
        self._fTau = sp.lambdify(args, tau, "numpy")

    # -- DynamicsEngine protocol -----------------------------------------
    def compute(self, model: RobotModel, state: State, gravity: float) -> DynOut:
        # Validate model: designed for 2R
        if model.dof != 2:
            raise ValueError("SympyLagrangeEngine currently supports only planar 2R models.")

        # link lengths (fallback 1.0)
        l1 = float(model.links[0].length or 1.0)
        l2 = float(model.links[1].length or 1.0)
        m1 = float(model.links[0].mass)
        m2 = float(model.links[1].mass)

        # COM fractions along each link (default to midpoint)
        def frac(link) -> float:
            if link.length and link.length != 0 and link.com is not None:
                return float(link.com[0] / link.length)
            return 0.5

        c1 = frac(model.links[0])
        c2 = frac(model.links[1])

        q1, q2 = map(float, state.q)
        qd1, qd2 = map(float, state.qd)
        if state.qdd is None:
            qdd1 = qdd2 = 0.0
        else:
            qdd1, qdd2 = map(float, state.qdd)

        args = (q1, q2, qd1, qd2, qdd1, qdd2, l1, l2, m1, m2, c1, c2, float(gravity))

        M = np.array(self._fM(*args), dtype=float)
        C = np.array(self._fC(*args), dtype=float)
        G = np.array(self._fG(*args), dtype=float).reshape(2)
        tau = np.array(self._fTau(*args), dtype=float).reshape(2)

        return DynOut(M=M, C=C, g=G, tau=tau)


# ======================================================================
# Optional Pinocchio Engine (stub that delegates to SymPy)
# ======================================================================

class PinocchioEngine:
    """
    Optional engine intended for high-performance rigid-body dynamics via
    Pinocchio. To keep the package importable even when Pinocchio is not
    installed (or not yet implemented), this stub delegates to the SymPy
    engine so the public API remains stable and tests pass.

    Replace `compute` with a real Pinocchio pipeline when ready.
    """
    def __init__(self) -> None:
        # Keep a working fallback
        self._fallback = SympyLagrangeEngine()

    def compute(self, model: RobotModel, state: State, gravity: float) -> DynOut:
        return self._fallback.compute(model, state, gravity)


__all__ = [
    "DynamicsEngine",
    "DynOut",
    "SympyLagrangeEngine",
    "PinocchioEngine",
]
