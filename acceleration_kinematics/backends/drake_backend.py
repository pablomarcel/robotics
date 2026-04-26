# acceleration_kinematics/backends/drake_backend.py
"""
Drake backend for **acceleration_kinematics kinematics**.

This adapter conforms to :class:`acceleration_kinematics.backends.base.Backend` and exposes
6D frame kinematics (angular_velocity on top, linear on bottom) using Drake's
MultibodyPlant:

    J(q)          ← CalcJacobianSpatialVelocity(..., JacobianWrtVariable.kV, ...)
    (J̇ q̇)(q, v) ← CalcBiasSpatialAcceleration(...)
    ẍ            ← J @ v̇ + (J̇ v)

Design choices
--------------
- Task dimension m = 6 for all frames (spatial quantities).
- We *express* Jacobians and accelerations in the **world** frame by default.
- For fixed-base arms (common in unit tests), nq == nv and you can pass joint
  arrays directly via `ChainState`. For floating-base models, use the `jacobian`
  and `jdot_qdot` methods with full q (nq) and v (nv) explicitly.

Optional dependency
-------------------
This module is optional. It raises an ImportError if `pydrake` is unavailable.

Examples
--------
>>> be = DrakeBackend.from_urdf("my_arm.urdf")  # world-expressed by default
>>> kin = ChainKinematics(backend=be, frame="tool0")
>>> J = be.jacobian("tool0", q)          # (6, nv)
>>> b = be.jdot_qdot("tool0", q, v)      # (6,)
>>> xdd = kin.forward_accel(q, v, vd)    # (6,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .base import Backend, ChainState, ArrayLike

# Drake import is **optional**; fail gracefully at runtime.
try:  # pragma: no cover (exercised only when Drake is installed)
    from pydrake.multibody.plant import MultibodyPlant
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.tree import JacobianWrtVariable
    _HAVE_DRAKE = True
except Exception:  # pragma: no cover
    MultibodyPlant = object  # type: ignore[misc,assignment]
    Parser = object          # type: ignore[misc,assignment]
    JacobianWrtVariable = object  # type: ignore[misc,assignment]
    _HAVE_DRAKE = False


def _ensure_drake():
    if not _HAVE_DRAKE:  # pragma: no cover
        raise ImportError("pydrake is not installed. Install Drake to use DrakeBackend.")


@dataclass
class DrakeBackend(Backend):
    """
    Drake-backed implementation of the `Backend` protocol.

    Parameters
    ----------
    plant : MultibodyPlant
        A finalized plant.
    context : plant.Context
        A default context for computations (this backend mutates it).
    expressed_in : str
        Name of the frame in which spatial quantities are expressed. Defaults
        to "world" (recommended). You can set any frame name resolvable by the plant.

    Notes
    -----
    - `dof()` returns plant.num_velocities().
    - `frames()` yields known frame names if `frame_names` is provided at init;
      otherwise returns just ("world",) while still accepting arbitrary frame
      names in API calls.
    """
    plant: MultibodyPlant
    context: "MultibodyPlant.Context"  # type: ignore[name-defined]
    expressed_in: str = "world"
    frame_names: Optional[Sequence[str]] = None

    # ----------------------- construction helpers -----------------------

    @classmethod
    def from_urdf(cls, urdf_path: str, *, time_step: float = 0.0,
                  expressed_in: str = "world",
                  package_dirs: Optional[Sequence[str]] = None,
                  frame_names: Optional[Sequence[str]] = None) -> "DrakeBackend":
        """
        Build a backend from a URDF file.

        Parameters
        ----------
        urdf_path : str
            Path to the URDF.
        time_step : float
            Plant time step (0.0 for continuous).
        expressed_in : str
            Frame name in which to express results (default "world").
        package_dirs : Optional[Sequence[str]]
            Additional package directories for mesh resolution (optional).
        frame_names : Optional[Sequence[str]]
            Optional list of frame names to expose via frames().

        Returns
        -------
        DrakeBackend
        """
        _ensure_drake()
        plant = MultibodyPlant(time_step)
        parser = Parser(plant)
        if package_dirs:
            for d in package_dirs:
                parser.package_map().PopulateFromFolder(d)
        parser.AddModelFromFile(urdf_path)
        plant.Finalize()
        context = plant.CreateDefaultContext()
        return cls(plant=plant, context=context, expressed_in=expressed_in, frame_names=frame_names)

    @classmethod
    def from_plant(cls, plant: MultibodyPlant, *, expressed_in: str = "world",
                   frame_names: Optional[Sequence[str]] = None) -> "DrakeBackend":
        _ensure_drake()
        if not plant.is_finalized():
            plant.Finalize()
        context = plant.CreateDefaultContext()
        return cls(plant=plant, context=context, expressed_in=expressed_in, frame_names=frame_names)

    # ----------------------- Protocol: discovery ------------------------

    def dof(self) -> int:
        _ensure_drake()
        return int(self.plant.num_velocities())

    def frames(self) -> Iterable[str]:
        # We don’t enumerate Drake’s internal frames unless the caller supplied
        # a curated list (keeps this optional backend lightweight/testable).
        return tuple(self.frame_names) if self.frame_names else ("world",)

    # ------------------------- internal checks -------------------------

    def _frame(self, name: str):
        _ensure_drake()
        try:
            return self.plant.GetFrameByName(name)
        except Exception as exc:
            raise ValueError(f"Unknown frame name: {name!r}") from exc

    def _expressed_in_frame(self):
        return self._frame(self.expressed_in)

    def _coerce_q(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1)
        nq = int(self.plant.num_positions())
        if q.size != nq:
            raise ValueError(f"q must have length nq={nq}, got {q.size}")
        return q

    def _coerce_v(self, v: ArrayLike) -> np.ndarray:
        v = np.asarray(v, float).reshape(-1)
        nv = int(self.plant.num_velocities())
        if v.size != nv:
            raise ValueError(f"q̇/q̈ must have length nv={nv}, got {v.size}")
        return v

    def _set_state(self, q: np.ndarray, v: Optional[np.ndarray] = None, vd: Optional[np.ndarray] = None):
        """Mutate self.context with (q, v); vd is only used for completeness."""
        self.plant.SetPositions(self.context, q)
        if v is not None:
            self.plant.SetVelocities(self.context, v)
        # Drake uses separate evaluation for accelerations; vd not stored on context.

    # --------------------------- core API ------------------------------

    def jacobian(self, frame: str, q: ArrayLike) -> np.ndarray:
        """
        Spatial-velocity_kinematics Jacobian J(q) ∈ R^{6×nv} of a frame point, expressed in `expressed_in`.
        We use the frame’s origin as the point of interest.
        """
        _ensure_drake()
        q = self._coerce_q(q)
        B = self._frame(frame)                  # frame B whose origin we track
        A = self._expressed_in_frame()          # express in frame A (default world)
        E = A                                   # measure position vector in A as well
        p_AoBo_A = np.zeros(3)                  # Bo at its origin → vector is zero

        self._set_state(q)
        J = self.plant.CalcJacobianSpatialVelocity(
            self.context, JacobianWrtVariable.kV, B, p_AoBo_A, A, E
        )  # shape (6, nv)
        return np.asarray(J, float)

    def jdot_qdot(self, frame: str, q: ArrayLike, qd: ArrayLike) -> np.ndarray:
        """
        Spatial bias term (J̇ v) ∈ R^6 for the frame origin, expressed in `expressed_in`.
        """
        _ensure_drake()
        q = self._coerce_q(q)
        v = self._coerce_v(qd)
        B = self._frame(frame)
        A = self._expressed_in_frame()
        E = A
        p_AoBo_A = np.zeros(3)

        self._set_state(q, v)
        bias = self.plant.CalcBiasSpatialAcceleration(
            self.context, JacobianWrtVariable.kV, B, p_AoBo_A, A, E
        )  # shape (6,)
        return np.asarray(bias, float).reshape(6,)

    def spatial_accel(self, frame: str, state: ChainState) -> np.ndarray:
        """
        Spatial acceleration_kinematics ẍ ∈ R^6 of the frame origin, expressed in `expressed_in`:

            ẍ = J(q) v̇ + (J̇ v)

        For fixed-base arms where nq == nv (typical), you can pass ChainState
        directly. For floating-base models, prefer calling `jacobian` and
        `jdot_qdot` with full-sized q and v explicitly to avoid ambiguity.
        """
        _ensure_drake()
        nq = int(self.plant.num_positions())
        nv = int(self.plant.num_velocities())
        if state.q.size != nq or state.qd.size != nv or state.qdd.size != nv:
            raise ValueError(
                f"State sizes must be (q:{nq}, qd:{nv}, qdd:{nv}); "
                f"got (q:{state.q.size}, qd:{state.qd.size}, qdd:{state.qdd.size})"
            )

        J = self.jacobian(frame, state.q)              # (6, nv)
        bias = self.jdot_qdot(frame, state.q, state.qd)  # (6,)
        xdd = J @ np.asarray(state.qdd, float).reshape(nv,) + bias
        return np.asarray(xdd, float).reshape(6,)
