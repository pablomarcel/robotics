# acceleration_kinematics/backends/base.py
"""
Backend protocol for **acceleration_kinematics kinematics**.

This module defines:
- `ChainState` : an immutable, shape-checked container for (q, qd, qdd)
- `Backend`    : a runtime-checkable Protocol that backends must implement

Why this shape?
---------------
`acceleration_kinematics.core.ChainKinematics` depends only on this interface:

    xdd = J(q) @ qdd + Jdot_qdot(q, qd)

where:
- J(q)            is an (m×n) task Jacobian for the chosen `frame`
- Jdot_qdot(q,qd) is the (m,) bias term
- spatial_accel   returns the same xdd an app/test would compute with J and Jdot_qdot

The protocol also exposes small discoverability helpers:
- `dof()`    → expected joint dimension (n)
- `frames()` → iterable of valid frame names (e.g., ["ee"])

These are lightweight but make unit tests clearer and CLI/service validation easier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Iterable, runtime_checkable, Union

import numpy as np


# ------------------------------- type aliases -------------------------------

ArrayLike = Union[Sequence[float], np.ndarray]


# -------------------------------- ChainState --------------------------------

@dataclass(frozen=True)
class ChainState:
    """
    Immutable, validated joint state.

    Attributes
    ----------
    q   : (n,) ndarray[float]
    qd  : (n,) ndarray[float]
    qdd : (n,) ndarray[float]

    Notes
    -----
    - Inputs are coerced to 1-D float arrays.
    - Lengths must match; otherwise a ValueError is raised.
    """
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray

    def __post_init__(self) -> None:
        q   = np.asarray(self.q, dtype=float).reshape(-1)
        qd  = np.asarray(self.qd, dtype=float).reshape(-1)
        qdd = np.asarray(self.qdd, dtype=float).reshape(-1)
        n = q.size
        if qd.size != n or qdd.size != n:
            raise ValueError(f"ChainState expects equal lengths, got q:{n}, qd:{qd.size}, qdd:{qdd.size}")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "qd", qd)
        object.__setattr__(self, "qdd", qdd)

    @property
    def n(self) -> int:
        """Number of joints (dimension of q)."""
        return int(self.q.size)

    @classmethod
    def from_lists(cls, q: ArrayLike, qd: ArrayLike, qdd: ArrayLike) -> "ChainState":
        """Convenience constructor with the same validation semantics."""
        return cls(np.asarray(q, float), np.asarray(qd, float), np.asarray(qdd, float))


# --------------------------------- Backend ----------------------------------

@runtime_checkable
class Backend(Protocol):
    """
    Runtime-checkable protocol for acceleration_kinematics backends.

    Required methods
    ----------------
    dof() -> int
        Return expected joint dimension n.

    frames() -> Iterable[str]
        Return an iterable of valid frame names (must include the default used
        in your design, typically "ee").

    jacobian(frame: str, q: ArrayLike) -> np.ndarray
        Return J(q) with shape (m, n) for the given frame.

    jdot_qdot(frame: str, q: ArrayLike, qd: ArrayLike) -> np.ndarray
        Return (J̇(q, q̇) q̇) with shape (m,).

    spatial_accel(frame: str, state: ChainState) -> np.ndarray
        Return ẍ with shape (m,), numerically equal to J @ qdd + J̇ q̇.

    Shape contracts (for tests)
    ---------------------------
    - Let n = self.dof(), m = task dimension for `frame`.
    - jacobian(...).shape      == (m, n)
    - jdot_qdot(...).shape     == (m,)
    - spatial_accel(...).shape == (m,)
    - All returned arrays are float dtype.
    """

    # ---- discovery helpers ----
    def dof(self) -> int: ...
    def frames(self) -> Iterable[str]: ...

    # ---- core kinematics ----
    def jacobian(self, frame: str, q: ArrayLike) -> np.ndarray: ...
    def jdot_qdot(self, frame: str, q: ArrayLike, qd: ArrayLike) -> np.ndarray: ...
    def spatial_accel(self, frame: str, state: ChainState) -> np.ndarray: ...


__all__ = ["ArrayLike", "ChainState", "Backend"]
