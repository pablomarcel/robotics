# applied_dynamics/backends/__init__.py
from __future__ import annotations

"""
Backend factory and convenience selectors.
"""

from typing import Optional
from .base import DynamicsBackend, BackendNotAvailable

def get_backend(prefer: Optional[str] = None) -> DynamicsBackend:
    """
    Pick the first available backend, or a specific one by name.
    Order: Drake → Pinocchio → RTB.
    """
    prefer = (prefer or "").lower()
    if prefer in ("drake", "pydrake"):
        from .drake_backend import DrakeBackend
        if DrakeBackend.available():
            return DrakeBackend()
        raise BackendNotAvailable("Drake (pydrake) not available")

    if prefer in ("pinocchio",):
        from .pinocchio_backend import PinocchioBackend
        if PinocchioBackend.available():
            return PinocchioBackend()
        raise BackendNotAvailable("Pinocchio not available")

    if prefer in ("rtb", "roboticstoolbox"):
        from .rtb_backend import RTBBackend
        if RTBBackend.available():
            return RTBBackend()
        raise BackendNotAvailable("Robotics Toolbox for Python not available")

    # auto pick
    try:
        from .drake_backend import DrakeBackend
        if DrakeBackend.available():
            return DrakeBackend()
    except Exception:
        pass
    try:
        from .pinocchio_backend import PinocchioBackend
        if PinocchioBackend.available():
            return PinocchioBackend()
    except Exception:
        pass
    try:
        from .rtb_backend import RTBBackend
        if RTBBackend.available():
            return RTBBackend()
    except Exception:
        pass

    raise BackendNotAvailable(
        "No numeric backend available. Install one of: pydrake, pin, roboticstoolbox-python."
    )
