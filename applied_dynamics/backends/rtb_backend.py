# applied_dynamics/backends/rtb_backend.py
from __future__ import annotations

"""
Robotics Toolbox for Python backend (RTB).

Capabilities
------------
- Build DH models for:
    * Simple pendulum (point mass at the tip)
    * Planar 2R (COM at mid-link for each link)
- Numeric queries:
    * Mass matrix M(q)
    * Bias vector C(q, qd) qd + g(q) via RNE
    * Energies: K from M(q), V from COM heights and gravity vector
- Tiny step() integrator: explicit Euler on M(q) qdd = tau - b

Notes
-----
- Gravity convention: we set world gravity to **[0, -g, 0]** so the 2D
  planar examples live in the x–y plane with gravity along −y.
- Potential energy: V = -Σ m_i * g_vec · p_i, where p_i is the world COM
  of link i (taking the link’s local COM `r` through fkine_all transforms).
"""

from dataclasses import dataclass
from typing import Any, Tuple, Optional
import numpy as np

from .base import (
    DynamicsBackend,
    BackendNotAvailable,
    _require,
    as1d,
    assert_shape,
    BackendRegistry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_rtb():
    try:
        import roboticstoolbox as rtb  # type: ignore
        from spatialmath import SE3    # type: ignore
        return rtb, SE3
    except Exception as exc:  # pragma: no cover - exercised only when missing
        raise BackendNotAvailable("roboticstoolbox (rtb) is not installed.") from exc


def _gravity_vec(g: float) -> np.ndarray:
    """Gravity along negative world-y."""
    return np.array([0.0, -float(g), 0.0], dtype=float)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class RTBBackend(DynamicsBackend):
    name: str = "rtb"

    # ------------- discovery -------------
    @staticmethod
    def available() -> bool:
        try:
            import roboticstoolbox as _  # noqa: F401
            return True
        except Exception:
            return False

    def _check(self) -> None:
        _require(self.available(), "roboticstoolbox (rtb) is not installed.")

    # ------------- builders -------------
    def build_simple_pendulum(self, m: float, l: float, g: float) -> Any:
        """
        One revolute about +z at the base, mass concentrated at the tip.
        DH: a=l, alpha=0, d=0
        COM: r = [l, 0, 0] in link frame (point mass at end)
        """
        self._check()
        rtb, _ = _import_rtb()
        L1 = rtb.RevoluteDH(a=float(l), alpha=0.0, d=0.0,
                            m=float(m), r=[float(l), 0.0, 0.0], G=0.0)
        robot = rtb.DHRobot([L1], name="simple_pendulum")
        robot.gravity = _gravity_vec(g)
        return robot

    def build_planar_2r(self, m1: float, m2: float, l1: float, l2: float, g: float) -> Any:
        """
        Two planar revolute joints about +z.
        COMs at mid-link: r = [a/2, 0, 0] for each link.
        """
        self._check()
        rtb, _ = _import_rtb()
        L1 = rtb.RevoluteDH(a=float(l1), alpha=0.0, d=0.0,
                            m=float(m1), r=[float(l1)/2.0, 0.0, 0.0], G=0.0)
        L2 = rtb.RevoluteDH(a=float(l2), alpha=0.0, d=0.0,
                            m=float(m2), r=[float(l2)/2.0, 0.0, 0.0], G=0.0)
        robot = rtb.DHRobot([L1, L2], name="planar2r")
        robot.gravity = _gravity_vec(g)
        return robot

    def build_cart_pendulum_absorber(self, M: float, m: float, l: float, k: float, g: float) -> Any:
        """
        Not supported natively in this RTB sample backend (cart prismatic + pendulum + spring).
        Could be approximated by custom dynamics, but we keep it explicit for clarity.
        """
        raise BackendNotAvailable("Cart–pendulum absorber is not implemented in RTB backend.")

    # ------------- numeric queries -------------

    def mass_matrix(self, model: Any, q: np.ndarray) -> np.ndarray:
        """
        M(q) from RTB's `inertia(q)`.
        """
        self._check()
        q = as1d(q)
        M = np.asarray(model.inertia(q), dtype=float)
        # RTB returns (n,n)
        return M

    def bias_coriolis_gravity(self, model: Any, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """
        τ = rne(q, qd, 0) = C(q, qd) qd + g(q)
        """
        self._check()
        q = as1d(q)
        qd = as1d(qd, n=q.size)
        tau = np.asarray(model.rne(q, qd, np.zeros_like(q)), dtype=float)
        tau = tau.reshape(-1)
        assert_shape(tau, q.size)
        return tau

    def energies(self, model: Any, q: np.ndarray, qd: np.ndarray) -> Tuple[float, float]:
        """
        K = 1/2 qd^T M(q) qd
        V = -Σ m_i g^T p_i, where p_i is world COM of link i.
        """
        self._check()
        q = as1d(q)
        qd = as1d(qd, n=q.size)

        # Kinetic via mass matrix
        M = self.mass_matrix(model, q)
        K = 0.5 * float(qd.reshape(1, -1) @ (M @ qd.reshape(-1, 1)))

        # Potential from COM heights and gravity vector
        gvec = np.asarray(getattr(model, "gravity", np.array([0, 0, -9.81])), dtype=float).reshape(3)
        # Forward kinematics to each link frame (end-of-link frames)
        Ts = model.fkine_all(q)  # list of SE3, one per link
        V = 0.0
        for i, L in enumerate(model.links):
            m_i = float(getattr(L, "m", 0.0))
            if m_i == 0.0:
                continue
            # Local COM in link frame
            r_i = np.asarray(getattr(L, "r", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            # World COM
            Ti = Ts[i]  # SE3
            p_i = (Ti.A @ np.r_[r_i, 1.0])[:3]
            V += -m_i * float(gvec.dot(p_i))
        return float(K), float(V)

    def step(self, model: Any, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Explicit Euler step on: M(q) qdd = tau - (C(q,qd)qd + g(q))
        """
        self._check()
        q = as1d(q)
        qd = as1d(qd, n=q.size)
        tau = as1d(tau, n=q.size)
        M = self.mass_matrix(model, q)
        b = self.bias_coriolis_gravity(model, q, qd)
        qdd = np.linalg.solve(M, tau - b)
        qd1 = qd + float(dt) * qdd
        q1 = q + float(dt) * qd1
        return q1, qd1


# Register with the backend registry so `select_backend()` can find us.
BackendRegistry.register(RTBBackend)
