# applied/models.py
from __future__ import annotations

"""
Applied Dynamics models (Eq. 10.1–10.398), OO + TDD friendly.

Each model:
- Implements the System & Energy protocols from applied.core
- Inherits ModelBase to get:
  * q, qd, dof               → test-friendly interface
  * lagrangian()             → returns SymPy expression L(q, qd, t)
  * energy(q=None, qd=None)  → returns {"K": K, "V": V} (symbolic or numeric)
  * equations_of_motion()    → Lagrange EOM vector (symbolic)

New models added beyond the initial scaffold:
- Trebuchet                 (10.342–10.346)
- EMParticle                (10.388–10.392)
- VehicleOnEarth (lat/lon)  (spherical surface kinematics)

Notes
-----
- For numeric evaluation, pass arrays/lists into energy(q, qd) — the method
  substitutes {qi(t): q[i], dqi(t): qd[i]} into K and V.
- The default generalized forces are zero; override generalized_forces() if needed.
"""

import sympy as sp
from dataclasses import dataclass
from typing import Dict, Tuple, List

from .core import FrameState, Energy, System

t = sp.symbols('t', real=True)  # global time symbol used by all models


# ---------------------------------------------------------------------------
# Base mixin: standardizes q/qd/dof, L, energy, EOM — keeps models crisp.
# ---------------------------------------------------------------------------

class ModelBase(System, Energy):
    """
    Mixin that provides:
      - properties q, qd, dof
      - lagrangian(): K - V
      - energy(q=None, qd=None): {"K": K, "V": V}
      - equations_of_motion(): Lagrange EOMs (symbolic)
    Models only need to implement:
      - lagrangian_state() -> (q_vec, qd_vec, t)
      - kinetic(fs) and potential(fs)
      - generalized_forces(fs)  (default: zeros)
    """

    # ---------- shape/introspection ----------

    @property
    def q(self) -> sp.Matrix:
        q, _, _ = self.lagrangian_state()
        return sp.Matrix(q)

    @property
    def qd(self) -> sp.Matrix:
        _, qd, _ = self.lagrangian_state()
        return sp.Matrix(qd)

    @property
    def dof(self) -> int:
        return int(self.q.shape[0])

    # ---------- energies / Lagrangian ----------

    def energy(self, q: List[float] | sp.Matrix | None = None,
               qd: List[float] | sp.Matrix | None = None) -> Dict[str, sp.Expr | float]:
        """
        Return {"K": K, "V": V}. If q and qd are provided, numerically substitute.
        """
        q_sym, qd_sym, _ = self.lagrangian_state()
        fs = FrameState(sp.Matrix(q_sym), sp.Matrix(qd_sym))

        K = sp.simplify(self.kinetic(fs))
        V = sp.simplify(self.potential(fs))

        if q is None or qd is None:
            return {"K": K, "V": V}

        # numeric substitution: { qi(t)->q[i], dqi(t)->qd[i] }
        subs = {}
        q_vals = list(q)
        qd_vals = list(qd)
        for i, qi in enumerate(q_sym):
            subs[qi] = q_vals[i]
        for i, qdi in enumerate(qd_sym):
            subs[qdi] = qd_vals[i]
        return {"K": float(K.subs(subs)), "V": float(V.subs(subs))}

    def lagrangian(self) -> sp.Expr:
        q_sym, qd_sym, _ = self.lagrangian_state()
        fs = FrameState(sp.Matrix(q_sym), sp.Matrix(qd_sym))
        K = sp.simplify(self.kinetic(fs))
        V = sp.simplify(self.potential(fs))
        return sp.simplify(K - V)

    # ---------- default generalized forces ----------

    def generalized_forces(self, fs: FrameState) -> sp.Matrix:
        return sp.Matrix.zeros(self.dof, 1)

    # ---------- EOM via Lagrange engine (no circular import at module import) ----------

    def equations_of_motion(self) -> sp.Matrix:
        """
        Build Lagrange EOMs symbolically for this model.

        Returns
        -------
        sp.Matrix
            Column vector of Euler–Lagrange equations (size = dof).
        """
        from .dynamics import LagrangeEngine  # local import to avoid cycles

        # Symbols/state
        q_sym, qd_sym, t_sym = self.lagrangian_state()

        # Recover the Function constructors used to define q(t)
        # e.g., theta(t) -> theta ; fallback to q1, q2, ...
        q_funcs: List[sp.Function] = []
        for idx, expr in enumerate(q_sym, start=1):
            if isinstance(expr, sp.AppliedUndef):
                q_funcs.append(expr.func)  # theta(t) -> theta
            else:
                q_funcs.append(sp.Function(f"q{idx}"))

        # Energies with the same symbolic state
        fs = FrameState(sp.Matrix(q_sym), sp.Matrix(qd_sym))
        K = sp.simplify(self.kinetic(fs))
        V = sp.simplify(self.potential(fs))

        # Generalized forces (ensure a (n,1) Matrix; default zeros if None/empty)
        Q = self.generalized_forces(fs)
        if Q is None:
            Q = sp.Matrix.zeros(len(q_sym), 1)
        else:
            Q = sp.Matrix(Q)
            if Q.shape == (len(q_sym),):
                Q = Q.reshape(len(q_sym), 1)
            assert Q.shape == (len(q_sym), 1), "generalized_forces must be a column vector of size dof"

        # Lagrange engine
        engine = LagrangeEngine()
        return engine.equations_of_motion(q_funcs, t_sym, K, V, Q)


# ===========================================================================
# Concrete models
# ===========================================================================

# ----------------------------- Simple Pendulum ------------------------------

@dataclass
class SimplePendulum(ModelBase):
    """
    Simple pendulum with angle θ about pivot; bob mass at distance l.
    """
    m: sp.Symbol; l: sp.Symbol; g: sp.Symbol
    th = sp.Function('theta')

    def lagrangian_state(self) -> Tuple[sp.Matrix, sp.Matrix, sp.Symbol]:
        q = sp.Matrix([self.th(t)])
        qd = sp.Matrix([sp.diff(self.th(t), t)])
        return q, qd, t

    # Energy protocol
    def kinetic(self, fs: FrameState) -> sp.Expr:
        (theta,) = fs.q
        (thetad,) = fs.qd
        return sp.Rational(1, 2) * self.m * (self.l**2) * (thetad**2)

    def potential(self, fs: FrameState) -> sp.Expr:
        (theta,) = fs.q
        return self.m * self.g * self.l * (1 - sp.cos(theta))


# --------------------------- Spherical Pendulum -----------------------------

@dataclass
class SphericalPendulum(ModelBase):
    """
    Spherical pendulum with angles (θ, φ) and constant rod length l.
    """
    m: sp.Symbol; l: sp.Symbol; g: sp.Symbol
    th = sp.Function('theta'); ph = sp.Function('phi')

    def lagrangian_state(self) -> Tuple[sp.Matrix, sp.Matrix, sp.Symbol]:
        q = sp.Matrix([self.th(t), self.ph(t)])
        qd = sp.Matrix([sp.diff(self.th(t), t), sp.diff(self.ph(t), t)])
        return q, qd, t

    def kinetic(self, fs: FrameState) -> sp.Expr:
        th, ph = fs.q
        thd, phd = fs.qd
        return sp.Rational(1, 2) * self.m * (self.l**2) * (thd**2 + (phd**2) * (sp.sin(th) ** 2))

    def potential(self, fs: FrameState) -> sp.Expr:
        (th, _) = fs.q
        return self.m * self.g * self.l * sp.cos(th)


# ------------------------------- Planar 2R ----------------------------------

@dataclass
class Planar2R(ModelBase):
    """
    Planar 2R manipulator with point masses at link ends (compact textbook form).
    Generalized coordinates: (θ1, θ2)
    """
    m1: sp.Symbol; m2: sp.Symbol; l1: sp.Symbol; l2: sp.Symbol; g: sp.Symbol
    th1 = sp.Function('theta1'); th2 = sp.Function('theta2')

    def lagrangian_state(self) -> Tuple[sp.Matrix, sp.Matrix, sp.Symbol]:
        q  = sp.Matrix([self.th1(t), self.th2(t)])
        qd = sp.Matrix([sp.diff(self.th1(t), t), sp.diff(self.th2(t), t)])
        return q, qd, t

    def kinetic(self, fs: FrameState) -> sp.Expr:
        th1, th2 = fs.q
        d1, d2 = fs.qd
        K1 = sp.Rational(1, 2) * self.m1 * (self.l1**2) * (d1**2)
        K2 = sp.Rational(1, 2) * self.m2 * (
            (self.l2**2 + self.l1**2 + 2*self.l1*self.l2*sp.cos(th2)) * (d1**2)
            + (self.l2**2) * (d2**2)
            + 2*self.l1*self.l2*sp.cos(th2) * d1*d2
        )
        return sp.simplify(K1 + K2)

    def potential(self, fs: FrameState) -> sp.Expr:
        th1, th2 = fs.q
        return self.m1*self.g*self.l1*sp.sin(th1) + self.m2*self.g*(self.l1*sp.sin(th1) + self.l2*sp.sin(th1 + th2))


# ------------------------ Cart–Pendulum Absorber ----------------------------

@dataclass
class CartPendulumAbsorber(ModelBase):
    """
    Vibration absorber: cart of mass M with a pendulum bob (m, l) and spring k.
    Generalized coordinates: (x, θ)
    """
    M: sp.Symbol; m: sp.Symbol; l: sp.Symbol; k: sp.Symbol; g: sp.Symbol
    x = sp.Function('x'); th = sp.Function('theta')

    def lagrangian_state(self) -> Tuple[sp.Matrix, sp.Matrix, sp.Symbol]:
        q  = sp.Matrix([self.x(t), self.th(t)])
        qd = sp.Matrix([sp.diff(self.x(t), t), sp.diff(self.th(t), t)])
        return q, qd, t

    def kinetic(self, fs: FrameState) -> sp.Expr:
        x, th = fs.q
        xd, thd = fs.qd
        # textbook compact expression
        return sp.Rational(1, 2)*self.M*(xd**2) + sp.Rational(1, 2)*self.m*(xd**2 + (self.l**2)*(thd**2) + 2*self.l*xd*thd*sp.cos(th))

    def potential(self, fs: FrameState) -> sp.Expr:
        x, th = fs.q
        return sp.Rational(1, 2)*self.k*(x**2) + self.m*self.g*self.l*(1 - sp.cos(th))


# ------------------------------- Trebuchet ----------------------------------

@dataclass
class Trebuchet(ModelBase):
    """
    Minimal planar trebuchet (per 10.342–10.346 simplification):

    q = (θ, α, y)
      θ : main arm angle wrt +x (counterclockwise)
      α : secondary arm (sling) relative angle
      y : vertical slider offset for projectile (positive up)

    Parameters
    ----------
    m1 : mass of counterweight
    m2 : mass of projectile
    a  : pivot→hinge offset along main arm
    b  : hinge→counterweight offset along secondary link
    l  : main arm tip length (pivot→projectile attach)
    g  : gravity (positive scalar)

    Kinematics (2D x–y plane; y upward):
      M  = O + a [cos θ, sin θ]
      P1 = M + b [cos(θ+α), sin(θ+α)]           (counterweight position)
      P2 = M - l [cos θ, sin θ] + y [0, 1]      (projectile position)
    """
    m1: sp.Symbol; m2: sp.Symbol
    a: sp.Symbol; b: sp.Symbol; l: sp.Symbol
    g: sp.Symbol
    th = sp.Function('theta'); al = sp.Function('alpha'); y = sp.Function('y')

    def lagrangian_state(self) -> Tuple[sp.Matrix, sp.Matrix, sp.Symbol]:
        q  = sp.Matrix([self.th(t), self.al(t), self.y(t)])
        qd = sp.Matrix([sp.diff(self.th(t), t), sp.diff(self.al(t), t), sp.diff(self.y(t), t)])
        return q, qd, t

    def _positions(self) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
        """Return (M, P1, P2) planar position vectors in N{x,y}."""
        th, al, y = self.th(t), self.al(t), self.y(t)
        M  = sp.Matrix([ self.a*sp.cos(th),  self.a*sp.sin(th) ])
        P1 = sp.Matrix([ self.b*sp.cos(th+al), self.b*sp.sin(th+al) ]) + M
        P2 = sp.Matrix([ -self.l*sp.cos(th), -self.l*sp.sin(th) ]) + M + sp.Matrix([0, y])
        return M, P1, P2

    def _velocities(self) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
        M, P1, P2 = self._positions()
        Md  = sp.diff(M, t)
        P1d = sp.diff(P1, t)
        P2d = sp.diff(P2, t)
        return Md, P1d, P2d

    def kinetic(self, fs: FrameState) -> sp.Expr:
        _, P1d, P2d = self._velocities()
        K = sp.Rational(1, 2)*self.m1*(P1d.dot(P1d)) + sp.Rational(1, 2)*self.m2*(P2d.dot(P2d))
        return sp.simplify(K)

    def potential(self, fs: FrameState) -> sp.Expr:
        _, P1, P2 = (None,)+self._positions()[1:]  # ignore M
        V = self.m1*self.g*P1[1] + self.m2*self.g*P2[1]  # y-components
        return sp.simplify(V)


# ----------------------------- EM Lagrangian --------------------------------

@dataclass
class EMParticle(ModelBase):
    """
    Charged particle in an electromagnetic field (10.388–10.392):

    L = 1/2 m ||v||^2 - e φ(t, x, y, z) + e v·A(t, x, y, z)

    Coordinates
    -----------
    q = (x, y, z)

    Field potentials (symbolic by default)
    --------------------------------------
    φ = Phi(t, x, y, z)
    A = (A1(t, x, y, z), A2(...), A3(...))
    """
    m: sp.Symbol; e: sp.Symbol
    Phi = sp.Function('Phi')
    A1  = sp.Function('A1'); A2 = sp.Function('A2'); A3 = sp.Function('A3')

    x = sp.Function('x'); y = sp.Function('y'); z = sp.Function('z')

    def lagrangian_state(self) -> Tuple[sp.Matrix, sp.Matrix, sp.Symbol]:
        q  = sp.Matrix([self.x(t), self.y(t), self.z(t)])
        qd = sp.Matrix([sp.diff(self.x(t), t), sp.diff(self.y(t), t), sp.diff(self.z(t), t)])
        return q, qd, t

    def _potentials(self) -> Tuple[sp.Expr, sp.Matrix]:
        x, y, z = self.x(t), self.y(t), self.z(t)
        phi = self.Phi(t, x, y, z)
        A   = sp.Matrix([self.A1(t, x, y, z), self.A2(t, x, y, z), self.A3(t, x, y, z)])
        return phi, A

    def kinetic(self, fs: FrameState) -> sp.Expr:
        v = sp.Matrix(fs.qd)
        return sp.Rational(1, 2) * self.m * (v.dot(v))

    def potential(self, fs: FrameState) -> sp.Expr:
        # In the Lagrangian formalism, V is the scalar potential contribution (+ convention),
        # but our energy() returns physical potential energy; keep V = e*phi here so L=K - V + e v·A.
        phi, _ = self._potentials()
        return self.e * phi

    def lagrangian(self) -> sp.Expr:
        q_sym, qd_sym, _ = self.lagrangian_state()
        fs = FrameState(q_sym, qd_sym)
        K = self.kinetic(fs)
        phi, A = self._potentials()
        v = sp.Matrix(qd_sym)
        # L = K - e*phi + e v·A
        return sp.simplify(K - self.e*phi + self.e * (v.dot(A)))


# -------------------------- Vehicle on Earth (lat/lon) ----------------------

@dataclass
class VehicleOnEarth(ModelBase):
    """
    Point-mass constrained to a spherical Earth of radius R (constant altitude).

    Coordinates: q = (λ, φ)  [latitude λ, longitude φ]
    Kinetic energy (standard spherical metric):
      K = 1/2 m R^2 ( λ̇^2 + (cos λ)^2 φ̇^2 )
    Potential: constant (set V = 0). For small altitude variations, extend V.

    This is a clean testbed for non-Cartesian coordinates (10.359–10.364).
    """
    m: sp.Symbol; R: sp.Symbol
    la = sp.Function('lat')   # λ
    lo = sp.Function('lon')   # φ

    def lagrangian_state(self) -> Tuple[sp.Matrix, sp.Matrix, sp.Symbol]:
        q  = sp.Matrix([self.la(t), self.lo(t)])
        qd = sp.Matrix([sp.diff(self.la(t), t), sp.diff(self.lo(t), t)])
        return q, qd, t

    def kinetic(self, fs: FrameState) -> sp.Expr:
        la, lo = fs.q
        lad, lod = fs.qd
        return sp.Rational(1, 2) * self.m * (self.R**2) * (lad**2 + (sp.cos(la)**2) * (lod**2))

    def potential(self, fs: FrameState) -> sp.Expr:
        return sp.Integer(0)
