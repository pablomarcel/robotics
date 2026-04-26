# time_optimal_control/app.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from .core import TimeOptimalProblem, SolveResult
from .design import Planar2RGeom  # retained for future extensions

# -------- Optional imports guarded --------
_casadi = None
try:
    import casadi as ca
    _casadi = ca
except Exception:
    pass

try:
    import toppra as ta
    import toppra.constraint as constraint
    import toppra.algorithm as algo
except Exception:
    ta = None
    constraint = None
    algo = None


# =======================
# 1) Double-integrator OCP (free-final-time), optional CasADi
# =======================

class MinTimeDoubleIntegrator(TimeOptimalProblem):
    """
    Minimize tf subject to:
      ẋ = v
      v̇ = u/m - μ g - c v  (Coulomb friction μ and linear drag c optional)
      |u| <= F
    """
    def __init__(self, name: str, x0: float, xf: float, m: float = 1.0, F: float = 10.0,
                 mu: float = 0.0, drag: float = 0.0, g: float = 9.81):
        super().__init__(name)
        self.x0, self.xf = float(x0), float(xf)
        self.m, self.F = float(m), float(F)
        self.mu, self.drag, self.g = float(mu), float(drag), float(g)
        self.N = 80  # discretization intervals
        self._sol: Optional[Dict[str, Any]] = None
        self._bounds = None
        self._nlp = None
        self._solver = None

    def build(self):
        if _casadi is None:
            raise RuntimeError("CasADi is required for MinTimeDoubleIntegrator.")
        ca = _casadi
        N = self.N

        # Decision variables
        tf = ca.MX.sym("tf")
        h = tf / N
        x = ca.MX.sym("x"); v = ca.MX.sym("v"); u = ca.MX.sym("u")
        f = ca.vertcat(v, u/self.m - self.mu*self.g - self.drag*v)
        F_dyn = ca.Function("F", [x, v, u], [f])

        X = [ca.MX.sym(f"X_{k}", 2) for k in range(N+1)]   # [x,v] at nodes
        U = [ca.MX.sym(f"U_{k}") for k in range(N)]        # control_techniques per interval

        # Variable vector layout: [ tf, X_0(2), ..., X_N(2), U_0, ..., U_{N-1} ]
        w = [tf] + X + U
        xcat = ca.vertcat(*w)

        # Bounds arrays (dense)
        nx = 1 + 2*(N+1) + N
        lbx = -1e20 * np.ones(nx)
        ubx =  1e20 * np.ones(nx)

        # Helpers to locate indices in w
        def idx_X(k: int) -> int:  # start index for X_k in flattened w
            return 1 + 2*k
        def idx_U(k: int) -> int:
            return 1 + 2*(N+1) + k

        # tf bounds
        lbx[0], ubx[0] = 0.01, 10.0

        # Boundary conditions: x(0)=x0, v(0)=0; x(tf)=xf, v(tf)=0
        lbx[idx_X(0):idx_X(0)+2] = [self.x0, 0.0]
        ubx[idx_X(0):idx_X(0)+2] = [self.x0, 0.0]
        lbx[idx_X(N):idx_X(N)+2] = [self.xf, 0.0]
        ubx[idx_X(N):idx_X(N)+2] = [self.xf, 0.0]

        # Input bounds
        for k in range(N):
            lbx[idx_U(k)] = -self.F
            ubx[idx_U(k)] =  self.F

        # Dynamics constraints (forward_kinematics Euler for compactness)
        gcons = []
        lbg = []
        ubg = []
        for k in range(N):
            step = X[k] + h * F_dyn(X[k][0], X[k][1], U[k])
            gcons.append(X[k+1] - step)
            lbg += [0.0, 0.0]
            ubg += [0.0, 0.0]

        gcat = ca.vertcat(*gcons) if len(gcons) else ca.MX.zeros(0, 1)
        prob = {'f': tf, 'x': xcat, 'g': gcat}

        # Solver: Ipopt if available, else built-in SQP (no external deps)
        try:
            self._solver = ca.nlpsol("s", "ipopt", prob,
                                     {"ipopt.print_level": 0, "print_time": 0})
        except Exception:
            self._solver = ca.nlpsol("s", "sqpmethod", prob,
                                     {"print_time": 0, "max_iter": 500})

        self._nlp = prob
        self._bounds = (lbx, ubx, np.array(lbg), np.array(ubg))
        self._built = True
        return self

    def solve(self):
        ca = _casadi
        N = self.N
        lbx, ubx, lbg, ubg = self._bounds

        # Simple initial guess (tf=1, states/inputs zeros)
        w0 = np.zeros_like(lbx)
        w0[0] = 1.0
        # optional: linear interpolation for x to help convergence
        for k in range(N+1):
            alpha = k/float(N)
            w0[1 + 2*k] = (1 - alpha)*self.x0 + alpha*self.xf  # x guess

        sol = self._solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=w0)
        wopt = np.array(sol['x']).squeeze()
        tf = float(wopt[0])

        self._sol = {"tf": tf, "name": self.name}
        self._solved = True
        return self

    def result(self) -> SolveResult:
        data = {"ok": self._solved, "name": self.name, "tf": self._sol.get("tf", None)}
        return SolveResult(ok=self._solved, message="OK" if self._solved else "NOT_SOLVED", data=data)


# =======================
# 2) 2R path_planning time-parameterization via TOPPRA (0.6.3 API)
# =======================

@dataclass
class TwoRParams:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g:  float = 9.81
    tau_max: Tuple[float, float] = (100.0, 100.0)  # |P|≤, |Q|≤


class TwoRPathTimeScaler(TimeOptimalProblem):
    """
    Time-optimal path_planning parameterization along a given joint path_planning q(s) using TOPPRA.
    Matches §13.3 (Eqs. 13.129–13.146) and Fig. 13.14/13.16 behavior.
    """
    def __init__(self, name: str, qs: np.ndarray, params: TwoRParams):
        super().__init__(name)
        if ta is None:
            raise RuntimeError("TOPPRA is required (import toppra failed).")
        self.qs = np.asarray(qs, float)  # (N,2)
        self.params = params
        self._algo = None
        self._traj = None
        self._grid = None
        self._sol_pack: Dict[str, Any] = {}

    # --- simple rigid 2R model pieces ---
    def _twoR_inertia(self, q: np.ndarray) -> np.ndarray:
        p = self.params
        th, ph = q
        a11 = p.m1*(p.l1**2)/3 + p.m2*(p.l1**2 + (p.l2**2)/3 + p.l1*p.l2*np.cos(ph))
        a22 = p.m2*(p.l2**2)/3
        a12 = p.m2*((p.l2**2)/3 + 0.5*p.l1*p.l2*np.cos(ph))
        return np.array([[a11, a12], [a12, a22]], dtype=float)

    def _twoR_gravity(self, q: np.ndarray) -> np.ndarray:
        p = self.params
        th, ph = q
        g1 = (p.m1*p.l1/2 + p.m2*p.l1)*p.g*np.cos(th) + p.m2*p.l2/2*p.g*np.cos(th+ph)
        g2 = p.m2*p.l2/2*p.g*np.cos(th+ph)
        return np.array([g1, g2], dtype=float)

    def _inv_dyn(self, q, qd, qdd):
        """
        TOPPRA 0.6.3 callback: must return (2,) for single sample, (N,2) for batch.
        Minimal ID: tau = D(q) qdd + G(q)  (Coriolis omitted for simplicity).
        """
        q = np.asarray(q)
        qd = np.asarray(qd)
        qdd = np.asarray(qdd)

        # Single sample: 1-D inputs of shape (2,)
        if q.ndim == 1:
            D = self._twoR_inertia(q)
            G = self._twoR_gravity(q)
            return D @ qdd + G  # shape (2,)

        # Batch: shape (N, 2)
        N = q.shape[0]
        tau = np.zeros((N, 2), dtype=float)
        for i in range(N):
            D = self._twoR_inertia(q[i])
            G = self._twoR_gravity(q[i])
            tau[i] = D @ qdd[i] + G
        return tau

    def _build_toppra_constraints(self):
        # Torque limits matrix (dof, 2): columns [min, max]
        umax = np.array(self.params.tau_max, dtype=float)
        umin = -umax
        tau_lim = np.vstack([umin, umax]).T  # [[minP,maxP],[minQ,maxQ]]

        # Dry-friction coefficients per joint (use zeros unless you want stick-slip)
        fs_coef = np.zeros(2, dtype=float)

        # 0.6.3 signature: (inv_dyn, tau_lim, fs_coef, discretization_scheme=?)
        scheme = getattr(constraint, "DiscretizationType", None)
        scheme = scheme.Interpolation if scheme is not None else 1  # Interpolation recommended

        torq_con = constraint.JointTorqueConstraint(self._inv_dyn, tau_lim, fs_coef, scheme)
        return [torq_con]

    def build(self):
        constraints = self._build_toppra_constraints()
        path = ta.SplineInterpolator(np.linspace(0, 1, len(self.qs)), self.qs)
        self._algo = algo.TOPPRA(constraints, path, solver_wrapper="seidel")
        self._built = True
        return self

    def solve(self):
        # toppra==0.6.3: compute_trajectory returns a ParametrizeSpline-like object
        traj = self._algo.compute_trajectory(0.0, 0.0)
        tf = float(traj.get_duration())

        # For test reporting and convenience, create a simple time grid to sample later if needed
        tgrid = np.linspace(0.0, tf, 129)  # 128 intervals -> 129 points

        self._traj = traj
        self._grid = tgrid
        self._sol_pack = {
            "tf": tf,
            "grid_size": int(len(tgrid)),
            "q0": self.qs[0].tolist(),
            "qf": self.qs[-1].tolist(),
            "name": self.name,
        }
        self._solved = True
        return self

    def result(self) -> SolveResult:
        return SolveResult(True, "OK", self._sol_pack)
