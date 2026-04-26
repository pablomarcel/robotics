# applied/backends/pinocchio_backend.py
from __future__ import annotations

"""
Pinocchio backend.

Capabilities
------------
- Build compact models (RZ joints) for:
    * Simple pendulum (point mass at the tip)
    * Planar 2R (COMs at mid-links)
- Query numerics:
    * M(q) via CRBA
    * bias b(q,qd) = C(q,qd) qd + g(q) via RNEA
    * Energies: K from M(q), V from COM heights using explicit COM frames
- Tiny explicit-Euler step()

Gravity convention
------------------
We set gravity along negative world-y: g_vec = [0, -g, 0].
Pinocchio stores gravity as a spatial motion_kinematics in `model.gravity`; we set its
linear part to that vector (angular_velocity part zero).
"""

from dataclasses import dataclass
from typing import Any, Tuple, Optional, List
import numpy as np

from .base import (
    DynamicsBackend,
    BackendNotAvailable,
    _require,
    BackendRegistry,
    as1d,
    assert_shape,
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _import_pin():
    try:
        import pinocchio as pin  # type: ignore
        return pin
    except Exception as exc:  # pragma: no cover
        raise BackendNotAvailable("pinocchio is not installed.") from exc


def _gvec(g: float) -> np.ndarray:
    """Gravity along −y (planar examples in x–y)."""
    return np.array([0.0, -float(g), 0.0], dtype=float)


def _set_gravity(model, g: float) -> None:
    """Set model.gravity to have zero angular_velocity part and linear part = _gvec(g)."""
    pin = _import_pin()
    gv = _gvec(g)
    # Motion is (angular_velocity, linear)
    model.gravity = pin.Motion(np.r_[np.zeros(3), gv])


def _add_com_frame(pin, model, joint_id: int, placement, name: str) -> None:
    """
    Add a Frame at the COM placement so we can retrieve its world position later.
    """
    # Use BODY frame type; placement is SE3 in the joint frame
    model.addFrame(pin.Frame(name, joint_id, placement, pin.FrameType.BODY))


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class PinocchioBackend(DynamicsBackend):
    name: str = "pinocchio"

    # ------------- discovery -------------

    @staticmethod
    def available() -> bool:
        try:
            import pinocchio as _  # noqa: F401
            return True
        except Exception:
            return False

    def _check(self) -> None:
        _require(self.available(), "pinocchio is not installed.")

    # ------------- builders -------------

    def build_simple_pendulum(self, m: float, l: float, g: float) -> Any:
        """
        One RZ joint at the base, point mass m at the *tip* (offset l along +x).
        COM is captured via a dedicated Frame 'COM_j{jid}' for potential energy.
        """
        self._check()
        pin = _import_pin()

        model = pin.Model()
        data = model.createData()

        # Universe → j1 (RZ)
        jz = pin.JointModelRZ()
        j1 = model.addJoint(model.getJointId("universe"), jz, pin.SE3.Identity(), "j1")

        # Inertia: sphere at origin of the COM body frame; we *place* it at x=l.
        inertia = pin.Inertia.FromSphere(float(m), 1e-6)
        X_J_to_COM = pin.SE3(np.eye(3), np.array([float(l), 0.0, 0.0]))
        model.appendBodyToJoint(j1, inertia, X_J_to_COM)

        # Add a named frame at the COM
        _add_com_frame(pin, model, j1, X_J_to_COM, f"COM_j{j1}")

        # Gravity
        _set_gravity(model, g)
        return (model, data)

    def build_planar_2r(self, m1: float, m2: float, l1: float, l2: float, g: float) -> Any:
        """
        Two RZ joints in series. COMs placed at mid-links [a/2,0,0] for each link.
        Named frames: 'COM_j{j1}', 'COM_j{j2}'.
        """
        self._check()
        pin = _import_pin()

        model = pin.Model()
        data = model.createData()

        # Universe → j1 (RZ)
        j1 = model.addJoint(model.getJointId("universe"), pin.JointModelRZ(), pin.SE3.Identity(), "j1")

        # Link-1 inertia at mid-link
        I1 = pin.Inertia.FromSphere(float(m1), 1e-6)
        X1 = pin.SE3(np.eye(3), np.array([float(l1) / 2.0, 0.0, 0.0]))
        model.appendBodyToJoint(j1, I1, X1)
        _add_com_frame(pin, model, j1, X1, f"COM_j{j1}")

        # j1 → j2 (RZ) located at x = l1
        j2 = model.addJoint(j1, pin.JointModelRZ(), pin.SE3(np.eye(3), np.array([float(l1), 0.0, 0.0])), "j2")

        # Link-2 inertia at mid-link
        I2 = pin.Inertia.FromSphere(float(m2), 1e-6)
        X2 = pin.SE3(np.eye(3), np.array([float(l2) / 2.0, 0.0, 0.0]))
        model.appendBodyToJoint(j2, I2, X2)
        _add_com_frame(pin, model, j2, X2, f"COM_j{j2}")

        # Gravity
        _set_gravity(model, g)
        return (model, data)

    def build_cart_pendulum_absorber(self, M: float, m: float, l: float, k: float, g: float) -> Any:
        """
        Not implemented in this sample backend (requires mixed prismatic/revolute build).
        """
        raise BackendNotAvailable("Cart–pendulum absorber is not implemented in Pinocchio backend.")

    # ------------- numeric queries -------------

    def mass_matrix(self, model: Any, q: np.ndarray) -> np.ndarray:
        """
        M(q) via CRBA.
        """
        self._check()
        pin = _import_pin()
        mdl, data = model
        q = as1d(q, n=mdl.nq)
        pin.crba(mdl, data, q)
        # data.M is symmetric; return as ndarray
        return np.asarray(data.M, dtype=float)

    def bias_coriolis_gravity(self, model: Any, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """
        b(q, qd) = C(q,qd) qd + g(q) via RNEA with qdd=0.
        """
        self._check()
        pin = _import_pin()
        mdl, data = model
        q = as1d(q, n=mdl.nq)
        qd = as1d(qd, n=mdl.nv)
        b = pin.rnea(mdl, data, q, qd, np.zeros_like(qd))
        b = np.asarray(b, dtype=float).reshape(-1)
        assert_shape(b, mdl.nv)
        return b

    def energies(self, model: Any, q: np.ndarray, qd: np.ndarray) -> Tuple[float, float]:
        """
        K = 1/2 qdᵀ M(q) qd
        V = - Σ_i m_i * g_vec · p_i(q), where p_i are COM world positions.
            (We added explicit COM frames 'COM_j{k}' in the builders.)
        """
        self._check()
        pin = _import_pin()
        mdl, data = model
        q = as1d(q, n=mdl.nq)
        qd = as1d(qd, n=mdl.nv)

        # Kinetic
        M = self.mass_matrix(model, q)
        K = 0.5 * float(qd.reshape(1, -1) @ (M @ qd.reshape(-1, 1)))

        # Potential from COM frames
        pin.forwardKinematics(mdl, data, q)
        pin.updateJointPlacements(mdl, data)
        pin.updateFramePlacements(mdl, data)

        gvec = _gvec( np.linalg.norm(mdl.gravity.linear) if hasattr(mdl.gravity, "linear") else 9.81 )

        V = 0.0
        # Iterate over joints (skip 0/universe)
        for jid in range(1, mdl.njoints):
            name = f"COM_j{jid}"
            try:
                fid = mdl.getFrameId(name)
            except Exception:
                continue  # frame not present for this joint
            p_i = np.asarray(data.oMf[fid].translation, dtype=float).reshape(3)
            # Retrieve mass attached to joint (we constructed one per joint)
            m_i = float(mdl.inertias[jid].mass)
            if m_i == 0.0:
                continue
            V += -m_i * float(gvec.dot(p_i))

        return float(K), float(V)

    def step(self, model: Any, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Explicit Euler on: M(q) qdd = tau - b(q, qd)
        """
        self._check()
        mdl, _ = model
        q = as1d(q, n=mdl.nq)
        qd = as1d(qd, n=mdl.nv)
        tau = as1d(tau, n=mdl.nv)

        M = self.mass_matrix(model, q)
        b = self.bias_coriolis_gravity(model, q, qd)
        qdd = np.linalg.solve(M, tau - b)
        qd1 = qd + float(dt) * qdd
        q1 = q + float(dt) * qd1
        return q1, qd1


# Register so select_backend() can discover us
BackendRegistry.register(PinocchioBackend)
