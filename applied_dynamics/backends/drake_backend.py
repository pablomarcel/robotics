# applied_dynamics/backends/drake_backend.py
from __future__ import annotations

"""
Drake backend (pydrake), import-safe version.

Key points
----------
- No top-level `from pydrake.all import ...` (prevents IDE/static errors when Drake isn't installed).
- Lazy loader `_pd()` brings in `pydrake.all` only inside methods.
- All public method signatures use `Any` for model objects to avoid referencing Drake types.
- Returns lightweight tuples with the plant + a few bits of metadata.
"""

from dataclasses import dataclass
from typing import Any, Tuple
import importlib
import numpy as np

from .base import DynamicsBackend, BackendNotAvailable, _require


def _pd():
    """
    Lazy import for `pydrake.all`. Raises BackendNotAvailable if not installed.
    """
    try:
        return importlib.import_module("pydrake.all")
    except Exception as e:
        raise BackendNotAvailable("pydrake is not installed") from e


@dataclass
class DrakeBackend(DynamicsBackend):
    name: str = "drake"

    # -------- discovery --------
    @staticmethod
    def available() -> bool:
        try:
            importlib.import_module("pydrake.all")
            return True
        except Exception:
            return False

    # -------- builders ---------
    def build_simple_pendulum(self, m: float, l: float, g: float) -> Any:
        pd = _pd()
        plant = pd.MultibodyPlant(0.0)

        # Rigid body with tiny rotational inertia; COM at origin (we'll use an offset for the bob location)
        inertia = pd.SpatialInertia(
            mass=m,
            p_PScm_E=np.zeros(3),
            G_SP_E=m * pd.UnitInertia.SolidSphere(1e-6),
        )
        body = plant.AddRigidBody("bob", inertia)

        # Revolute about +Z at the world origin
        plant.AddJoint(
            pd.RevoluteJoint(
                "hinge",
                plant.world_body(),
                body,
                np.eye(3),
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            )
        )

        # We'll keep the geometric offset in metadata (so energies can consider the length if needed in the future)
        X_BQ = np.eye(4)
        X_BQ[0, 3] = float(l)  # bob located at (l, 0, 0) in body frame

        plant.Finalize()
        # Return a tuple the rest of the backend understands
        return (plant, {"type": "pendulum", "X_BQ": X_BQ, "g": float(g)})

    def build_planar_2r(self, m1: float, m2: float, l1: float, l2: float, g: float) -> Any:
        pd = _pd()
        plant = pd.MultibodyPlant(0.0)

        b1 = plant.AddRigidBody(
            "link1",
            pd.SpatialInertia(m1, np.zeros(3), m1 * pd.UnitInertia.SolidSphere(1e-6)),
        )
        b2 = plant.AddRigidBody(
            "link2",
            pd.SpatialInertia(m2, np.zeros(3), m2 * pd.UnitInertia.SolidSphere(1e-6)),
        )

        # Revolute joints about +Z; j2 is placed at x=l1 of link1
        plant.AddJoint(
            pd.RevoluteJoint(
                "j1", plant.world_body(), b1,
                np.eye(3), [0, 0, 0], [0, 0, 0], [0, 0, 1]
            )
        )
        plant.AddJoint(
            pd.RevoluteJoint(
                "j2", b1, b2,
                np.eye(3), [float(l1), 0, 0], [0, 0, 0], [0, 0, 1]
            )
        )

        plant.Finalize()
        meta = {"type": "planar2r", "l1": float(l1), "l2": float(l2), "g": float(g)}
        return (plant, meta)

    def build_cart_pendulum_absorber(self, M: float, m: float, l: float, k: float, g: float) -> Any:
        pd = _pd()
        plant = pd.MultibodyPlant(0.0)

        cart = plant.AddRigidBody(
            "cart",
            pd.SpatialInertia(M, np.zeros(3), M * pd.UnitInertia.SolidBox(1e-3, 1e-3, 1e-3)),
        )
        bob = plant.AddRigidBody(
            "bob",
            pd.SpatialInertia(m, np.zeros(3), m * pd.UnitInertia.SolidSphere(1e-6)),
        )

        # Prismatic joint along X for the cart
        plant.AddJoint(
            pd.PrismaticJoint(
                "slider",
                plant.world_body(),
                cart,
                np.eye(3), [0, 0, 0], [0, 0, 0],
                axis=[1, 0, 0],
            )
        )
        # Revolute about +Z; bob frame offset by l on the cart
        plant.AddJoint(
            pd.RevoluteJoint(
                "hinge",
                cart, bob,
                np.eye(3), [0, 0, 0], [float(l), 0, 0],
                [0, 0, 1],
            )
        )

        plant.Finalize()
        meta = {"type": "cart_pendulum_absorber", "k": float(k), "g": float(g)}
        return (plant, meta)

    # -------- numeric queries --------
    def _check(self): _require(self.available(), "pydrake is not installed")

    def _context_set(self, plant: Any, q: np.ndarray | None, qd: np.ndarray | None):
        pd = _pd()
        context = plant.CreateDefaultContext()
        if q is not None:
            plant.SetPositions(context, np.asarray(q, dtype=float).reshape((-1,)))
        if qd is not None:
            plant.SetVelocities(context, np.asarray(qd, dtype=float).reshape((-1,)))
        return pd, context

    def mass_matrix(self, model: Any, q: np.ndarray) -> np.ndarray:
        self._check()
        plant, _ = model
        _, context = self._context_set(plant, q, None)
        M = plant.CalcMassMatrixViaInverseDynamics(context)
        return np.asarray(M)

    def bias_coriolis_gravity(self, model: Any, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        self._check()
        plant, meta = model
        _, context = self._context_set(plant, q, qd)
        tau_bias = plant.CalcBiasTerm(context)  # equals C(q,qd)qd + g(q)
        tau = np.asarray(tau_bias).reshape((-1,))

        # (Optional) add simple cart spring for the absorber example
        if meta.get("type") == "cart_pendulum_absorber" and q.size >= 1:
            k = float(meta["k"])
            x = float(q[0])
            tau = tau.copy()
            tau[0] -= k * x  # spring force along the prismatic joint

        return tau

    def energies(self, model: Any, q: np.ndarray, qd: np.ndarray) -> Tuple[float, float]:
        self._check()
        plant, _ = model
        _, context = self._context_set(plant, q, qd)
        K = float(plant.CalcKineticEnergy(context))
        V = float(plant.CalcPotentialEnergy(context))
        return K, V

    # -------- tiny sim --------
    def step(self, model: Any, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Semi-implicit Euler using M(q) and bias term b(q,qd) = Cqd + g.
        """
        M = self.mass_matrix(model, q)
        b = self.bias_coriolis_gravity(model, q, qd)
        a = np.linalg.solve(M, np.asarray(tau, dtype=float).reshape((-1,)) - b)
        qd1 = qd + dt * a
        q1 = q + dt * qd1
        return q1, qd1
