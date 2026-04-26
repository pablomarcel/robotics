# acceleration_kinematics/design.py
"""
Preset builders for **acceleration_kinematics-kinematics** workflows.

These presets return :class:`acceleration_kinematics.core.ChainKinematics` instances
so callers can immediately compute **forward_kinematics** and **inverse_kinematics** acceleration_kinematics
(ẍ = J q̈ + J̇ q̇  and  q̈ = J⁺(ẍ − J̇ q̇)) in a consistent, testable way.

What this module provides (lightweight, dependency-free):
- `planar_2r(l1, l2)` — a NumPy-backed 2R planar chain façade
- `from_backend(backend, frame="ee")` — wrap any Backend into ChainKinematics

Optional (future-friendly) stubs:
- `from_pinocchio(...)` / `from_drake(...)` — adapters if you add those backends

Design notes
------------
* We intentionally **do not** re-implement DH link objects here; that belongs in
  the inverse_kinematics-kinematics package. Acceleration workflows only need a `Backend`
  providing J, J̇q̇, and FK-derived quantities — already abstracted by
  :class:`acceleration_kinematics.backends.base.Backend`.
* Presets keep names/frames minimal (`frame="ee"`), which is easy to test.
"""

from __future__ import annotations

from typing import Optional

from .core import ChainKinematics
from .backends.base import Backend
from .backends.numpy_backend import Planar2R


# ------------------------------ Utilities --------------------------------- #

def from_backend(backend: Backend, *, frame: str = "ee") -> ChainKinematics:
    """
    Wrap a concrete :class:`Backend` into a :class:`ChainKinematics` façade.

    Parameters
    ----------
    backend : Backend
        Concrete implementation (NumPy, Pinocchio, Drake, etc.)
    frame : str
        Frame name (backend-dependent; "ee" by convention for end-effector).

    Returns
    -------
    ChainKinematics
    """
    return ChainKinematics(backend=backend, frame=frame)


# ------------------------------- Planar 2R -------------------------------- #

def planar_2r(l1: float, l2: float, *, name: str = "planar_2R") -> ChainKinematics:
    """
    Build a **planar 2R** chain façade backed by the NumPy backend.

    Geometry
    --------
    Standard planar arm in the x–y plane (z out of plane). The NumPy backend
    implements analytic expressions for:
      - `jacobian(q)` (2×2 planar Jacobian for end-effector XY)
      - `jdot_qdot(q, qd)` (planar bias term for XY)
      - `spatial_accel(frame, state)` → ẍ = J q̈ + J̇ q̇ in XY

    Parameters
    ----------
    l1, l2 : float
        Link lengths.

    Returns
    -------
    ChainKinematics
        Drop-in façade for forward_kinematics/inverse_kinematics acceleration_kinematics unit tests.
    """
    backend = Planar2R(float(l1), float(l2))
    # `name` is kept for API symmetry with inverse_kinematics/design, but ChainKinematics
    # does not currently store it; the backend may use it internally if needed.
    _ = name  # reserved / no-op to keep signature consistent and testable
    return from_backend(backend, frame="ee")


# ------------------------- Optional Heavy Backends ------------------------ #
# These are intentionally optional so unit tests run without heavy deps.
# You can implement the adapter backends (pinocchio_backend.py / drake_backend.py)
# and then uncomment/extend these helpers.

def from_pinocchio(model_or_path: object, *, frame: str = "ee") -> ChainKinematics:  # pragma: no cover
    """
    Create a ChainKinematics façade from a Pinocchio model/URDF/path.

    Requires:
        acceleration_kinematics.backends.pinocchio_backend.PinocchioBackend

    Notes
    -----
    This function is a stub to keep the design forward_kinematics-compatible without
    introducing a hard dependency on Pinocchio in unit tests.
    """
    try:
        from .backends.pinocchio_backend import PinocchioBackend  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Pinocchio backend is not available in this build.") from exc
    backend = PinocchioBackend(model_or_path)
    return from_backend(backend, frame=frame)


def from_drake(plant_or_builder: object, *, frame: str = "ee") -> ChainKinematics:  # pragma: no cover
    """
    Create a ChainKinematics façade from a Drake MultibodyPlant/builder.

    Requires:
        acceleration_kinematics.backends.drake_backend.DrakeBackend
    """
    try:
        from .backends.drake_backend import DrakeBackend  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Drake backend is not available in this build.") from exc
    backend = DrakeBackend(plant_or_builder)
    return from_backend(backend, frame=frame)


# --------------------------- Convenience Aliases -------------------------- #

build_planar_2r = planar_2r
build_from_backend = from_backend
build_from_pinocchio = from_pinocchio  # pragma: no cover (optional)
build_from_drake = from_drake          # pragma: no cover (optional)


__all__ = [
    "from_backend",
    "planar_2r",
    "from_pinocchio",
    "from_drake",
    "build_planar_2r",
    "build_from_backend",
    "build_from_pinocchio",
    "build_from_drake",
]
