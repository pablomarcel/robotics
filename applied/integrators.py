# applied/integrators.py
"""
Numeric simulation layer for Applied Dynamics.

Features
--------
- Symbolic → numeric adapter for Lagrangian systems:
    Extracts M(q) and bias b(q, qd, t) from Euler–Lagrange equations symbolically.
- Fast RHS via sympy.lambdify
- ODESolver facade:
    * Uses scipy.integrate.solve_ivp when available
    * Falls back to pure-Python RK4 (fixed step) if SciPy is absent
- Events / early termination (solve_ivp path), minimal event emulation in RK4
- Trajectory container + CSV export

Usage
-----
>>> from applied.design import DesignLibrary
>>> from applied.integrators import LagrangeRHS, ODESolver, IntegratorConfig
>>> lib = DesignLibrary()
>>> sys = lib.create("pendulum_num")                 # SimplePendulum with numeric params
>>> rhs = LagrangeRHS.from_model(sys)                # build \dot y = f(t,y)
>>> cfg = IntegratorConfig(t_span=(0.0, 5.0), rtol=1e-7, atol=1e-9)
>>> y0 = rhs.pack_state(q=[0.2], qd=[0.0])           # initial state vector
>>> sol = ODESolver(cfg).solve(rhs, y0)              # integrate
>>> len(sol.t), sol.y.shape

Notes
-----
State vector y is concatenated [q, qd]. For n-DoF systems, y has length 2n.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict, Any
import math
import numpy as np
import sympy as sp

from .core import FrameState
from .dynamics import LagrangeEngine
from .utils import OUT_DIR

# SciPy is optional
try:  # pragma: no cover
    from scipy.integrate import solve_ivp  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    solve_ivp = None
    _HAVE_SCIPY = False


# =============================================================================
# Symbolic second-order system adapter (Lagrange → first-order RHS)
# =============================================================================

@dataclass
class LagrangeRHS:
    """
    Adapter that turns a symbolic Lagrangian model into a numeric RHS ydot = f(t,y).

    It computes Euler–Lagrange equations, treats them as *linear in accelerations*,
    extracts M(q) and b(q, qd, t) via Jacobians, and lambdifies:
        qdd(q, qd, t) = - M(q)^{-1} b(q, qd, t)
        ydot = [qd; qdd]

    Attributes
    ----------
    q_syms : sp.Matrix
        [q1(t), ..., qn(t)]
    qd_syms : sp.Matrix
        [dq1/dt, ..., dqn/dt]
    t_sym : sp.Symbol
        time symbol used by the model
    acc_syms : sp.Matrix
        stand-in symbols [a1, ..., an] to replace d2qi/dt2 during linearization
    M_expr : sp.Matrix
        Symbolic mass matrix M(q)
    b_expr : sp.Matrix
        Symbolic bias vector b(q, qd, t)
    qdd_lambda : Callable
        Fast numeric function (q, qd, t) -> qdd ndarray
    """

    q_syms: sp.Matrix
    qd_syms: sp.Matrix
    t_sym: sp.Symbol
    acc_syms: sp.Matrix
    M_expr: sp.Matrix
    b_expr: sp.Matrix
    qdd_lambda: Callable[[np.ndarray, np.ndarray, float], np.ndarray]

    # --------------------------- construction --------------------------- #

    @staticmethod
    def _extract_mass_bias_from_eom(eom: sp.Matrix, q_syms: sp.Matrix, t_sym: sp.Symbol) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
        """
        From Euler–Lagrange equations e(q,qd,qdd,t)=0, return:
            (acc_syms, M(q), b(q,qd,t))   with e = M*a + b
        where `a` are placeholder symbols for qdd.
        """
        n = int(q_syms.shape[0])
        # Build qd & qdd symbols as they appear in the equations
        qd_syms = sp.Matrix([sp.diff(qi, t_sym) for qi in q_syms])
        qdd_syms = [sp.diff(qi, t_sym, 2) for qi in q_syms]
        a = sp.Matrix(sp.symbols("a1:%d" % (n+1)))  # a1..an

        # Replace qdd with 'a' so the system is linear in 'a'
        subs_qdd = {qdd: a[i] for i, qdd in enumerate(qdd_syms)}
        e_a = sp.Matrix([sp.simplify(expr.subs(subs_qdd)) for expr in eom])

        # Mass matrix M = ∂e/∂a ; Bias b = e|_{a=0}
        M = e_a.jacobian(a)
        b = sp.Matrix([sp.simplify(expr.subs({ai: 0 for ai in a})) for expr in e_a])

        return a, sp.simplify(M), sp.simplify(b)

    @classmethod
    def from_model(cls, model) -> "LagrangeRHS":
        """
        Build adapter from a model implementing:
            - lagrangian_state()
            - kinetic(fs), potential(fs), generalized_forces(fs)
        """
        q_syms, qd_syms, t_sym = model.lagrangian_state()
        fs = FrameState(sp.Matrix(q_syms), sp.Matrix(qd_syms))
        K = sp.simplify(model.kinetic(fs))
        V = sp.simplify(model.potential(fs))
        Q = model.generalized_forces(fs)
        if Q is None:
            Q = sp.Matrix.zeros(len(q_syms), 1)
        else:
            Q = sp.Matrix(Q)
            if Q.shape == (len(q_syms),):
                Q = Q.reshape(len(q_syms), 1)

        # Euler–Lagrange equations
        engine = LagrangeEngine()
        # We need the *function constructors*, e.g., theta for theta(t)
        q_funcs: List[sp.Function] = []
        for qi in q_syms:
            if isinstance(qi, sp.AppliedUndef):
                q_funcs.append(qi.func)
            else:
                q_funcs.append(sp.Function(f"q{len(q_funcs)+1}"))
        eom = engine.equations_of_motion(q_funcs, t_sym, K, V, Q)  # column vector

        # Extract M and b
        acc_syms, M, b = cls._extract_mass_bias_from_eom(eom, sp.Matrix(q_syms), t_sym)

        # Lambdify qdd = -M^{-1} b; use array-aware lambdas
        # Build numeric input list ordering
        q_list  = list(q_syms)
        qd_list = list(qd_syms)

        # create a function that evaluates M and b given (q, qd, t)
        inputs = q_list + qd_list + [t_sym]
        M_func = sp.lambdify(inputs, M, modules=["numpy"])
        B_func = sp.lambdify(inputs, b, modules=["numpy"])

        def qdd_lambda(q: np.ndarray, qd: np.ndarray, t: float) -> np.ndarray:
            inp = list(np.asarray(q, dtype=float).ravel()) + \
                  list(np.asarray(qd, dtype=float).ravel()) + [float(t)]
            Mv = np.asarray(M_func(*inp), dtype=float)
            bv = np.asarray(B_func(*inp), dtype=float).reshape((-1, 1))
            # Solve M a + b = 0  → a = -M^{-1} b
            a = np.linalg.solve(Mv, -bv)
            return a.ravel()

        return cls(
            q_syms=sp.Matrix(q_syms),
            qd_syms=sp.Matrix(qd_syms),
            t_sym=t_sym,
            acc_syms=acc_syms,
            M_expr=M,
            b_expr=b,
            qdd_lambda=qdd_lambda,
        )

    # ---------------------------- public API ----------------------------- #

    @property
    def dof(self) -> int:
        return int(self.q_syms.shape[0])

    def pack_state(self, q: Sequence[float], qd: Sequence[float]) -> np.ndarray:
        q = np.asarray(q, dtype=float).ravel()
        qd = np.asarray(qd, dtype=float).ravel()
        assert q.size == self.dof and qd.size == self.dof, "pack_state expects len(q)=len(qd)=dof"
        return np.concatenate([q, qd], axis=0)

    def split_state(self, y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float).ravel()
        assert y.size == 2*self.dof, "state vector must be size 2*dof"
        return y[:self.dof], y[self.dof:]

    def f(self, t: float, y: Sequence[float]) -> np.ndarray:
        """First-order RHS: ydot = [qd; qdd(q,qd,t)]"""
        q, qd = self.split_state(y)
        qdd = self.qdd_lambda(q, qd, float(t))
        return np.concatenate([qd, qdd], axis=0)


# =============================================================================
# Solver façade (+ RK4 fallback)
# =============================================================================

@dataclass(frozen=True)
class IntegratorConfig:
    t_span: Tuple[float, float] = (0.0, 5.0)
    method: str = "RK45"         # used only if SciPy is available
    rtol: float = 1e-7
    atol: float = 1e-9
    max_step: Optional[float] = None
    dense_output: bool = False
    # RK4 fallback config
    rk4_dt: float = 1e-3
    # events: list of callables g(t,y) → float (terminate on sign change crossing zero)
    events: Optional[List[Callable[[float, np.ndarray], float]]] = None

@dataclass
class Trajectory:
    t: np.ndarray          # shape (N,)
    y: np.ndarray          # shape (2n, N)
    info: Dict[str, Any]

    def q(self) -> np.ndarray:
        n2, N = self.y.shape
        n = n2 // 2
        return self.y[:n, :]

    def qd(self) -> np.ndarray:
        n2, N = self.y.shape
        n = n2 // 2
        return self.y[n:, :]

    def to_csv(self, path: str | None = None) -> str:
        import csv
        if path is None:
            path = str((OUT_DIR / "trajectory.csv").as_posix())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            n2, N = self.y.shape
            header = ["t"] + [f"y{i+1}" for i in range(n2)]
            writer.writerow(header)
            for k in range(N):
                writer.writerow([self.t[k]] + list(self.y[:, k]))
        return path


class ODESolver:
    """
    Unified ODE solver façade.
      - Uses scipy.integrate.solve_ivp when available (recommended)
      - Otherwise uses a robust fixed-step RK4 for quick tests/CI
    """

    def __init__(self, config: IntegratorConfig = IntegratorConfig()) -> None:
        self.cfg = config

    # ----------------------- SciPy backend ----------------------- #

    def _solve_scipy(self, rhs: LagrangeRHS, y0: Sequence[float]) -> Trajectory:  # pragma: no cover (depends on SciPy)
        assert _HAVE_SCIPY and solve_ivp is not None
        t0, tf = self.cfg.t_span

        # Wrap events if any
        events = None
        if self.cfg.events:
            def _wrap(ev):
                return lambda t, y: float(ev(t, y))
            events = [_wrap(g) for g in self.cfg.events]
            for ev in events:
                ev.terminal = True  # stop on first root
                ev.direction = 0

        sol = solve_ivp(
            fun=rhs.f,
            t_span=(float(t0), float(tf)),
            y0=np.asarray(y0, dtype=float).ravel(),
            method=self.cfg.method,
            rtol=self.cfg.rtol,
            atol=self.cfg.atol,
            max_step=self.cfg.max_step,
            dense_output=self.cfg.dense_output,
            events=events,
            vectorized=False,
        )
        y = sol.y
        t = sol.t
        info = {
            "success": bool(sol.success),
            "message": str(sol.message),
            "nfev": int(getattr(sol, "nfev", -1)),
            "njev": int(getattr(sol, "njev", -1)),
            "nlu": int(getattr(sol, "nlu", -1)),
            "t_events": [np.asarray(te) for te in getattr(sol, "t_events", [])] if events else [],
        }
        return Trajectory(t=np.asarray(t), y=np.asarray(y), info=info)

    # ----------------------- RK4 fallback ------------------------ #

    def _solve_rk4(self, rhs: LagrangeRHS, y0: Sequence[float]) -> Trajectory:
        t0, tf = self.cfg.t_span
        dt = float(self.cfg.rk4_dt)
        if dt <= 0.0:
            raise ValueError("rk4_dt must be positive")

        # Build time grid with an exact final time step
        N = max(1, int(math.ceil((tf - t0) / dt)))
        t = np.linspace(t0, tf, N + 1)
        y = np.zeros((2*rhs.dof, N + 1), dtype=float)
        y[:, 0] = np.asarray(y0, dtype=float).ravel()

        def step(tk, yk, h):
            k1 = rhs.f(tk, yk)
            k2 = rhs.f(tk + 0.5*h, yk + 0.5*h*k1)
            k3 = rhs.f(tk + 0.5*h, yk + 0.5*h*k2)
            k4 = rhs.f(tk + h,     yk + h*k3)
            return yk + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # naive event emulation: stop if any event changes sign
        def _event_triggered(tk, tk1, yk, yk1) -> bool:
            if not self.cfg.events:
                return False
            for g in self.cfg.events:
                v0 = float(g(tk,  yk))
                v1 = float(g(tk1, yk1))
                if (v0 == 0.0) or (v1 == 0.0) or (v0 > 0) != (v1 > 0):
                    return True
            return False

        for k in range(N):
            h = t[k+1] - t[k]
            y_next = step(t[k], y[:, k], h)
            if _event_triggered(t[k], t[k+1], y[:, k], y_next):
                # truncate at the event boundary (no root finding in fallback)
                y = y[:, :k+2]
                t = t[:k+2]
                break
            y[:, k+1] = y_next

        info = {"success": True, "message": "RK4 completed", "backend": "rk4"}
        return Trajectory(t=t, y=y, info=info)

    # ----------------------- public API -------------------------- #

    def solve(self, rhs: LagrangeRHS, y0: Sequence[float]) -> Trajectory:
        """
        Integrate \dot y = f(t,y) over cfg.t_span starting at y0.
        """
        if _HAVE_SCIPY:  # pragma: no cover
            return self._solve_scipy(rhs, y0)
        return self._solve_rk4(rhs, y0)
