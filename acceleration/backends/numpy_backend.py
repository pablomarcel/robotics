# acceleration/backends/numpy_backend.py
"""
NumPy reference backends for **acceleration kinematics**.

Currently provided
------------------
- Planar2R: 2-link planar arm (end-effector XY task)
    x = l1 cos q1 + l2 cos(q1+q2)
    y = l1 sin q1 + l2 sin(q1+q2)

    J(q) =
      [ -l1 s1 - l2 s12   -l2 s12 ]
      [  l1 c1 + l2 c12    l2 c12 ]

    (J̇ q̇)(q, q̇) =
      [ -l1 c1 q1d^2 - l2 c12 (q1d+q2d)^2 ]
      [ -l1 s1 q1d^2 - l2 s12 (q1d+q2d)^2 ]

This backend is deterministic, dependency-light, and ideal for pytest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterable as _Iterable, Sequence

import numpy as np

from .base import Backend, ChainState, ArrayLike
from ..utils import jdot_qdot_fd


@dataclass(frozen=True)
class Planar2R(Backend):
    """
    NumPy backend for a planar 2R manipulator with an XY task.

    Parameters
    ----------
    l1, l2 : float
        Link lengths.

    Notes
    -----
    - The only supported frame is "ee" (end-effector XY).
    - Task dimension m = 2, joint dimension n = 2.
    - All methods return float ndarrays with shapes:
        J: (2,2),  J̇q̇: (2,),  ẍ: (2,)
    """
    l1: float
    l2: float
    _frame: str = "ee"
    _fd_check: bool = False   # set True in tests if you want FD sanity-checks

    # ----------------------- discovery helpers -----------------------

    def dof(self) -> int:
        return 2

    def frames(self) -> _Iterable[str]:
        return (self._frame,)

    # ------------------------ internal helpers -----------------------

    @staticmethod
    def _coerce_q(q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1)
        if q.size != 2:
            raise ValueError(f"Planar2R expects q of length 2, got {q.size}")
        return q

    @staticmethod
    def _coerce_qd(qd: ArrayLike) -> np.ndarray:
        qd = np.asarray(qd, float).reshape(-1)
        if qd.size != 2:
            raise ValueError(f"Planar2R expects qd of length 2, got {qd.size}")
        return qd

    def _check_frame(self, frame: str) -> None:
        if frame != self._frame:
            raise ValueError(f"Unsupported frame {frame!r}; supported: '{self._frame}'")

    # --------------------------- Jacobian ----------------------------

    def jacobian(self, frame: str, q: ArrayLike) -> np.ndarray:
        """
        Return the 2x2 end-effector XY Jacobian J(q).
        """
        self._check_frame(frame)
        q = self._coerce_q(q)
        q1, q2 = float(q[0]), float(q[1])
        l1, l2 = float(self.l1), float(self.l2)

        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)

        J = np.array(
            [
                [-l1 * s1 - l2 * s12, -l2 * s12],
                [ l1 * c1 + l2 * c12,  l2 * c12],
            ],
            dtype=float,
        )
        return J

    # ------------------------- bias term J̇q̇ ------------------------

    def jdot_qdot(self, frame: str, q: ArrayLike, qd: ArrayLike) -> np.ndarray:
        """
        Return (J̇(q, q̇)) q̇ as a 2-vector using the closed-form expression.

        Optionally (if `_fd_check` is True) validates the result with a
        directional finite-difference and raises if they differ too much.
        """
        self._check_frame(frame)
        q = self._coerce_q(q)
        qd = self._coerce_qd(qd)

        q1, q2 = float(q[0]), float(q[1])
        q1d, q2d = float(qd[0]), float(qd[1])
        l1, l2 = float(self.l1), float(self.l2)

        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)

        sumd = q1d + q2d

        # Closed-form bias
        x_bias = -l1 * c1 * (q1d ** 2) - l2 * c12 * (sumd ** 2)
        y_bias = -l1 * s1 * (q1d ** 2) - l2 * s12 * (sumd ** 2)
        bias = np.array([x_bias, y_bias], dtype=float)

        if self._fd_check:
            fd = jdot_qdot_fd(lambda x: self.jacobian(frame, x), q, qd)
            if not np.allclose(bias, fd, atol=1e-9, rtol=1e-9):
                raise AssertionError(f"Planar2R Jdot*qdot mismatch.\nclosed-form={bias}\nfd={fd}")
        return bias

    # ------------------------- spatial accel ẍ -----------------------

    def spatial_accel(self, frame: str, state: ChainState) -> np.ndarray:
        """
        Return end-effector XY acceleration:
            ẍ = J(q) q̈ + J̇(q, q̇) q̇
        """
        self._check_frame(frame)
        if state.n != 2:
            raise ValueError(f"Planar2R expects state.n==2, got {state.n}")

        J = self.jacobian(frame, state.q)
        bias = self.jdot_qdot(frame, state.q, state.qd)
        xdd = J @ state.qdd + bias
        return np.asarray(xdd, float).reshape(2,)
