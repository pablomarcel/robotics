# acceleration/backends/pinocchio_backend.py
"""
Pinocchio backend for **acceleration kinematics**.

This adapter conforms to :class:`acceleration.backends.base.Backend` and provides
6D frame kinematics (angular on top, linear on bottom) using Pinocchio.

Key choices
-----------
- Task dimension m = 6 for all frames (spatial twist/accel ordering).
- Reference frame is configurable: "world" | "local" | "local_world_aligned".
- q has length model.nq, qd/qdd have length model.nv (Pinocchio conventions).

Dependencies
------------
This module is optional. Import errors are raised at construction time if
`pinocchio` is not available.

Examples
--------
>>> # Build from URDF path
>>> be = PinocchioBackend.from_urdf("my_robot.urdf", reference="world")
>>> kin = ChainKinematics(backend=be, frame="tool0")
>>> J = be.jacobian("tool0", q)           # (6, nv)
>>> b = be.jdot_qdot("tool0", q, qd)      # (6,)
>>> xdd = kin.forward_accel(q, qd, qdd)   # (6,)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

from .base import Backend, ChainState, ArrayLike

# Pinocchio import is **optional**; fail gracefully at runtime.
try:  # pragma: no cover - covered by run-time paths in environments with pinocchio
    import pinocchio as pin
    _HAVE_PIN = True
except Exception:  # pragma: no cover
    pin = None     # type: ignore
    _HAVE_PIN = False


# ------------------------------- helpers ------------------------------------

_REF_MAP = {
    "world": "WORLD",
    "local": "LOCAL",
    "local_world_aligned": "LOCAL_WORLD_ALIGNED",
}

def _ref_enum(name: str):
    if not _HAVE_PIN:
        raise ImportError("Pinocchio is not installed. Install `pin` to use PinocchioBackend.")
    key = str(name).strip().lower()
    if key not in _REF_MAP:
        raise ValueError(f"reference must be one of {list(_REF_MAP)}, got {name!r}")
    return getattr(pin.ReferenceFrame, _REF_MAP[key])


def _ensure_pinocchio():
    if not _HAVE_PIN:
        raise ImportError("Pinocchio is not installed. Install `pin` to use PinocchioBackend.")


# ------------------------------- backend ------------------------------------

@dataclass
class PinocchioBackend(Backend):
    """
    Pinocchio-backed implementation of the `Backend` protocol.

    Parameters
    ----------
    model : pin.Model
        Pinocchio model instance.
    data : pin.Data
        Pinocchio data instance (will be mutated).
    reference : {"world","local","local_world_aligned"}
        Reference frame for Jacobians/accelerations. Defaults to "world".

    Notes
    -----
    - `dof()` returns model.nv.
    - `frames()` returns all frame names in the model (order = model.frames).
    - `jacobian(frame, q)` expects q with length model.nq.
    - `jdot_qdot(frame, q, qd)` expects q with length model.nq and qd with length model.nv.
    - `spatial_accel(frame, state)` expects ChainState with q size model.nq? No —
      ChainState stores joint-space arrays (n = model.nv); we take q from `state.q`
      as generalized position **if the model has nq == nv**. For floating-base models,
      pass `q_full` explicitly to jacobian/jdot_qdot or build a thin wrapper that
      holds both `q_full` and (v,a). For most fixed-base arms (common in these tests),
      nq == nv and this is fine.
    """
    model: "pin.Model"   # type: ignore[name-defined]
    data: "pin.Data"     # type: ignore[name-defined]
    reference: str = "world"

    # ----------------------- construction helpers -----------------------

    @classmethod
    def from_model(cls, model: "pin.Model", *, reference: str = "world") -> "PinocchioBackend":
        _ensure_pinocchio()
        data = model.createData()
        return cls(model=model, data=data, reference=reference)

    @classmethod
    def from_urdf(cls, urdf_path: str, *, reference: str = "world", package_dirs: Sequence[str] | None = None) -> "PinocchioBackend":
        """
        Build backend from a URDF path.

        Parameters
        ----------
        urdf_path : str
            Path to the robot URDF.
        reference : str
            Frame reference convention.
        package_dirs : list[str] | None
            Optional ROS-style package search paths for meshes.
        """
        _ensure_pinocchio()
        if package_dirs is None:
            model = pin.buildModelFromUrdf(urdf_path)
        else:
            model = pin.buildModelFromUrdf(urdf_path, package_dirs)
        data = model.createData()
        return cls(model=model, data=data, reference=reference)

    # ----------------------- Protocol: discovery ------------------------

    def dof(self) -> int:
        return int(self.model.nv)

    def frames(self) -> Iterable[str]:
        return (f.name for f in self.model.frames)

    # ------------------------- internal checks -------------------------

    def _frame_id(self, frame: str) -> int:
        fid = self.model.getFrameId(frame)
        if fid == self.model.nframes:
            raise ValueError(f"Unknown frame name: {frame!r}")
        return int(fid)

    def _coerce_q(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1)
        if q.size != self.model.nq:
            raise ValueError(f"q must have length model.nq={self.model.nq}, got {q.size}")
        return q

    def _coerce_v(self, v: ArrayLike) -> np.ndarray:
        v = np.asarray(v, float).reshape(-1)
        if v.size != self.model.nv:
            raise ValueError(f"q̇/q̈ must have length model.nv={self.model.nv}, got {v.size}")
        return v

    # --------------------------- core API ------------------------------

    def jacobian(self, frame: str, q: ArrayLike) -> np.ndarray:
        """
        Frame Jacobian J(q) ∈ R^{6×nv} in the configured reference frame.
        """
        _ensure_pinocchio()
        q = self._coerce_q(q)
        fid = self._frame_id(frame)
        ref = _ref_enum(self.reference)

        # Joint Jacobians and frame placements
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(self.model, self.data, fid, ref)  # (6, nv)
        return np.asarray(J, float)

    def jdot_qdot(self, frame: str, q: ArrayLike, qd: ArrayLike) -> np.ndarray:
        """
        Bias term (J̇ q̇) for the given frame in the configured reference.
        """
        _ensure_pinocchio()
        q = self._coerce_q(q)
        v = self._coerce_v(qd)
        fid = self._frame_id(frame)
        ref = _ref_enum(self.reference)

        # Time variation of joint Jacobians + frame placements
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        Jdot = pin.getFrameJacobianTimeVariation(self.model, self.data, fid, ref)  # (6, nv)
        bias = Jdot @ v
        return np.asarray(bias, float).reshape(6,)

    def spatial_accel(self, frame: str, state: ChainState) -> np.ndarray:
        """
        Spatial frame acceleration ẍ ∈ R^6 (angular ⊕ linear).

        Computes:
            ẍ = J(q) q̈ + J̇(q, q̇) q̇
        using Pinocchio’s forward_kinematics kinematics for (q, q̇, q̈) and then
        returning the frame acceleration expressed in the configured reference.
        """
        _ensure_pinocchio()
        # For fixed-base arms, nq == nv; otherwise ensure your state.q represents q.
        # If your robot has a floating base, pass the full q directly to jacobian/jdot_qdot.
        q = self._coerce_q(state.q if state.q.size == self.model.nq else np.r_[state.q])
        v = self._coerce_v(state.qd)
        a = self._coerce_v(state.qdd)
        fid = self._frame_id(frame)
        ref = _ref_enum(self.reference)

        # Forward kinematics up to acceleration
        pin.forwardKinematics(self.model, self.data, q, v, a)
        pin.updateFramePlacements(self.model, self.data)

        # Get spatial acceleration in desired reference
        a_frame = pin.getFrameAcceleration(self.model, self.data, fid, ref)  # motion_kinematics object (6,)
        # `a_frame.vector` is 6D: angular (ω̇) then linear (v̇ in frame conv.)
        a_vec = np.asarray(a_frame.vector, float).reshape(6,)

        # For consistency with J/Jdot bias, we could equivalently compute J@qdd + Jdot@qd:
        # J = self.jacobian(frame, q)
        # bias = self.jdot_qdot(frame, q, v)
        # a_check = J @ a + bias
        # (You can assert closeness in tests if desired.)

        return a_vec
