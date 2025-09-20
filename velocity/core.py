# velocity/core.py
"""
Core kinematics for the Velocity Kinematics Toolkit.

This module provides:
  - A DH-based robot model (`DHRobot`) with:
      * forward kinematics (Craig standard DH)
      * geometric Jacobian via Jacobian-generating vectors
      * analytic Jacobian (Euler-rate mapping; ZYX & ZXZ provided)
  - A minimal URDF shim (`URDFRobot`) that raises a clear error unless a URDF
    backend is installed; you can wire in urdfpy/pinocchio later.
  - A small `solvers` namespace with resolved-rates and Newton–Raphson IK.

Design notes
------------
- OOP & testability: deterministic methods, NumPy-only math.
- Standard DH (Craig):  Tz(d) · Rz(θ) · Tx(a) · Rx(α).
- Jacobian columns: for revolute i, [ k_i × (p_e - p_i) ; k_i ]; for prismatic i,
  [ k_i ; 0 ] — computed in the base frame.
- Analytic Jacobian maps ω→Euler rates (JA = [Jv; G^{-1} Jω]), falling back to
  geometric J near kinematic singularities of the Euler map.

You can extend this with screw-axis/PoE later if needed; the public API stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Local helpers (kept small; implemented in utils.py which will be provided next)
from .utils import (
    transl,
    trotz,
    trotx,
    mmul,
    skew,
    pinv_damped,
)

# ------------------------------- Data classes ---------------------------------


@dataclass(slots=True)
class JointDH:
    """
    Standard DH joint.

    Attributes
    ----------
    name : str
        Joint name.
    joint_type : str
        'R' (revolute) or 'P' (prismatic).
    alpha, a, d, theta : float
        DH parameters (Craig). Note θ and d contain the *offsets*; the
        configuration variable is added on top (θ+q for R, d+q for P).
    axis_local : np.ndarray
        Joint axis in its *own* frame (z-axis for standard DH).
    """
    name: str
    joint_type: str  # 'R' or 'P'
    alpha: float
    a: float
    d: float
    theta: float
    axis_local: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))

    def transform(self, q_i: float) -> np.ndarray:
        """
        Homogeneous transform {i-1}->i using Craig standard DH:

            T = Tz(d_i) · Rz(θ_i) · Tx(a_i) · Rx(α_i)

        where
            θ_i = θ_offset + q_i  (revolute),  d_i = d_offset + q_i (prismatic).
        """
        theta = self.theta + (q_i if self.joint_type.upper() == "R" else 0.0)
        di = self.d + (q_i if self.joint_type.upper() == "P" else 0.0)
        return mmul(transl(0, 0, di), trotz(theta), transl(self.a, 0, 0), trotx(self.alpha))


# ------------------------------ Robot abstractions ----------------------------


class _BaseRobot:
    """Common helpers; concrete subclasses implement `.from_spec()` construction."""

    def fk(self, q: np.ndarray) -> Dict[str, Any]:
        """Return a dict containing at least 'T_0e' (4x4) and 'frames' (list of 4x4)."""
        Ts, Tn = self._fk_all(q)
        return {"T_0e": Tn, "frames": Ts}

    # --- these must be provided by subclasses ---
    def _fk_all(self, q: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def jacobian_geometric(self, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian_analytic(self, q: np.ndarray, euler: str = "ZYX") -> np.ndarray:
        raise NotImplementedError


class DHRobot(_BaseRobot):
    """
    Serial manipulator defined by a list of standard-DH joints.

    Notes
    -----
    - Angles in radians; lengths in your consistent units.
    - `tool` is an SE(3) transform from link-n to end-effector (TCP).
    """

    def __init__(self, joints: List[JointDH], tool: Optional[np.ndarray] = None, name: Optional[str] = None):
        self.joints: List[JointDH] = joints
        self.tool: np.ndarray = (np.eye(4) if tool is None else np.asarray(tool, dtype=float))
        self.name = name or "dh_robot"

    # ------------------------------ construction ------------------------------

    @staticmethod
    def from_spec(spec: "RobotSpecLike") -> "DHRobot":
        """
        Construct from a DH dict (as parsed by I/O layer). Expected structure:

        {
          "name": "arm",
          "joints": [
            {"name":"j1","type":"R","alpha":0.0,"a":0.5,"d":0.0,"theta":0.0},
            ...
          ],
          "tool": {"xyz":[0,0,0]}  # optional
        }
        """
        data = spec.data if hasattr(spec, "data") else spec  # accept raw dict for tests
        joints = []
        for jd in data["joints"]:
            joints.append(
                JointDH(
                    name=str(jd.get("name", f"j{len(joints)+1}")),
                    joint_type=str(jd["type"]).upper(),
                    alpha=float(jd.get("alpha", 0.0)),
                    a=float(jd.get("a", 0.0)),
                    d=float(jd.get("d", 0.0)),
                    theta=float(jd.get("theta", 0.0)),
                )
            )
        tool = np.eye(4)
        tdesc = data.get("tool")
        if isinstance(tdesc, dict) and "xyz" in tdesc:
            x, y, z = (float(v) for v in tdesc["xyz"])
            tool = transl(x, y, z)
        elif isinstance(tdesc, (list, tuple, np.ndarray)):
            T = np.asarray(tdesc, dtype=float)
            if T.shape == (4, 4):
                tool = T
        return DHRobot(joints=joints, tool=tool, name=data.get("name"))

    # ----------------------------- kinematic ops ------------------------------

    def _fk_all(self, q: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        q = np.asarray(q, dtype=float).ravel()
        if q.size != len(self.joints):
            raise ValueError(f"q has length {q.size}, expected {len(self.joints)}")
        T = np.eye(4)
        frames: List[np.ndarray] = [T.copy()]  # base
        for ji, qi in zip(self.joints, q):
            T = T @ ji.transform(qi)
            frames.append(T.copy())
        # attach tool
        Tn = T @ self.tool
        frames[-1] = Tn
        return frames, Tn

    def _axes_and_origins(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        axes: List[np.ndarray] = []
        origins: List[np.ndarray] = []
        # frames includes base at index 0 and final TCP at -1; joint i lives at frames[i]
        for j, Ti in zip(self.joints, frames[:-1]):
            R = Ti[:3, :3]
            p = Ti[:3, 3]
            k = R @ j.axis_local  # DH z-axis expressed in base
            axes.append(k)
            origins.append(p)
        return axes, origins

    def jacobian_geometric(self, q: np.ndarray) -> np.ndarray:
        """
        6×n geometric Jacobian via Jacobian-generating vectors (base frame):

            J_i = [ k_i × (p_e - p_i) ; k_i ]  for revolute
                  [ k_i               ; 0   ]  for prismatic

        Matches the book’s construction (velocity kinematics).
        """
        frames, Tn = self._fk_all(np.asarray(q, dtype=float).ravel())
        pe = Tn[:3, 3]
        axes, origins = self._axes_and_origins(frames)
        n = len(self.joints)
        J = np.zeros((6, n), dtype=float)
        for i, (j, k_i, p_i) in enumerate(zip(self.joints, axes, origins)):
            if j.joint_type == "R":
                J[:3, i] = np.cross(k_i, pe - p_i)
                J[3:, i] = k_i
            else:
                J[:3, i] = k_i
                J[3:, i] = 0.0
        return J

    # -------------------------- Analytic Jacobians -----------------------------

    def jacobian_analytic(self, q: np.ndarray, euler: str = "ZYX") -> np.ndarray:
        """
        Analytic Jacobian J_A mapping q̇ -> [v; φ̇] where φ are Euler angles.

        JA = [ Jv ; G^{-1}(φ) Jω ], with φ extracted from the current EE rotation.
        Supported sequences: 'ZYX' (yaw-pitch-roll), 'ZXZ'. Falls back to geometric
        if the Euler-rate map is ill-conditioned.
        """
        euler = euler.upper()
        Jg = self.jacobian_geometric(q)
        _, Tn = self._fk_all(np.asarray(q, dtype=float).ravel())
        R = Tn[:3, :3]
        ok, Ginv = _euler_rate_map_inverse_from_R(R, euler)
        if not ok:
            return Jg  # graceful fallback near singular Euler maps
        return np.vstack([Jg[:3, :], Ginv @ Jg[3:, :]])


class URDFRobot(_BaseRobot):
    """
    Placeholder URDF-backed robot. You can wire urdfpy/pinocchio here later.
    For now, this class raises a clear error if used, to avoid silent misuse.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "URDFRobot requires a URDF backend (e.g., urdfpy/pinocchio). "
            "Integrate your preferred library and implement from_spec(), "
            "_fk_all(), jacobian_geometric(), jacobian_analytic()."
        )

    @staticmethod
    def from_spec(spec: "RobotSpecLike") -> "URDFRobot":
        return URDFRobot()  # pragma: no cover


# ------------------------------- Solver helpers -------------------------------


class solvers:
    """
    Small solver namespace (kept here to avoid extra files).
    Adds:
      - resolved_rates(J, Xdot, damping, weights)
      - newton_ik(robot, q0, x_target, ...)
    """

    @staticmethod
    def resolved_rates(
        J: np.ndarray,
        Xdot: np.ndarray,
        damping: Optional[float] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Solve q̇ from Ẋ = J q̇ using (weighted) damped least squares.

        If J is square and well-conditioned and no weights/damping are provided,
        solve directly. Otherwise:

            q̇ = Jᵀ (J Jᵀ + λ² I)^{-1} Ẋ      (unweighted DLS)

        Weighted least squares with diagonal W:

            minimize ||W^{1/2}(J q̇ - Ẋ)||²  ⇒ use J_w = W^{1/2} J, Ẋ_w = W^{1/2} Ẋ
        """
        J = np.asarray(J, dtype=float)
        Xdot = np.asarray(Xdot, dtype=float).reshape(-1)
        m, n = J.shape

        # Square, full-rank, and no weights/damping → direct solve
        if weights is None and (damping is None or damping == 0.0) and m == n:
            try:
                return np.linalg.solve(J, Xdot)
            except np.linalg.LinAlgError:
                pass  # fall through to DLS

        lam = float(damping) if damping is not None else 0.0
        if lam <= 0.0:
            # adaptive tiny damping to avoid numerical issues
            s = np.linalg.svd(J, compute_uv=False)
            lam = 1e-6 * (s.max() + 1.0)

        if weights is None:
            # Unweighted DLS
            return J.T @ np.linalg.inv(J @ J.T + (lam**2) * np.eye(m)) @ Xdot
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.size != m:
                raise ValueError(f"weights has length {w.size}, expected {m}")
            Wsqrt = np.diag(np.sqrt(w))
            Jw = Wsqrt @ J
            Xw = Wsqrt @ Xdot
            return Jw.T @ np.linalg.inv(Jw @ Jw.T + (lam**2) * np.eye(m)) @ Xw

    @staticmethod
    def newton_ik(
        robot: _BaseRobot,
        q0: np.ndarray,
        x_target: Mapping[str, Any],
        max_iter: int = 50,
        tol: float = 1e-8,
        weights: Optional[Sequence[float]] = None,
        euler: str = "ZYX",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Newton–Raphson inverse kinematics on pose x = [p; orientation].

        x_target:
          - position: key 'p' : (3,)
          - orientation: one of
              * 'R' : (3x3) rotation matrix
              * 'euler': {'seq': 'ZYX'|'ZXZ'..., 'angles': (3,) in radians}

        Returns
        -------
        (q_sol, info) where info includes {'iters','converged','res_norm'}
        """
        q = np.asarray(q0, dtype=float).ravel()
        p_des: Optional[np.ndarray] = None
        R_des: Optional[np.ndarray] = None

        if "p" in x_target:
            p_des = np.asarray(x_target["p"], dtype=float).reshape(3)
        if "R" in x_target:
            R_des = np.asarray(x_target["R"], dtype=float).reshape(3, 3)
        elif "euler" in x_target:
            ed = x_target["euler"]
            seq = str(ed.get("seq", euler)).upper()
            ang = np.asarray(ed["angles"], dtype=float).reshape(3)
            R_des = _R_from_euler(seq, ang)
        if R_des is None:
            # default to keep same orientation; only position IK
            # extract current orientation at q0
            _, Tn0 = robot._fk_all(q)
            R_des = Tn0[:3, :3]

        res_hist: List[float] = []
        converged = False

        for k in range(1, max_iter + 1):
            frames, Tn = robot._fk_all(q)
            R = Tn[:3, :3]
            p = Tn[:3, 3]

            # Pose error: spatial (base-frame) 6×1 twist-like vector [dp; w_err]
            dp = (p_des - p) if p_des is not None else (0.0 * p)
            # orientation error via small-angle approx from R^T R_des
            Re = R.T @ R_des
            w_err = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            xerr = np.r_[dp, w_err]

            res = float(np.linalg.norm(xerr))
            res_hist.append(res)
            if res < tol:
                converged = True
                break

            J = robot.jacobian_geometric(q)
            dq = solvers.resolved_rates(J, xerr, damping=None, weights=weights)
            q = q + dq

        info = {"iters": k, "converged": converged, "res_norm": res_hist[-1], "res_hist": res_hist}
        return q, info


# --------------------------- Euler utilities ----------------------------------


def _R_from_euler(seq: str, ang: np.ndarray) -> np.ndarray:
    """Build rotation from Euler angles (radians). Supports 'ZYX' and 'ZXZ'."""
    c = np.cos
    s = np.sin
    a, b, g = ang.tolist()
    seq = seq.upper()
    if seq == "ZYX":
        Rz = np.array([[c(a), -s(a), 0], [s(a), c(a), 0], [0, 0, 1]])
        Ry = np.array([[c(b), 0, s(b)], [0, 1, 0], [-s(b), 0, c(b)]])
        Rx = np.array([[1, 0, 0], [0, c(g), -s(g)], [0, s(g), c(g)]])
        return Rz @ Ry @ Rx
    elif seq == "ZXZ":
        Rz1 = np.array([[c(a), -s(a), 0], [s(a), c(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, c(b), -s(b)], [0, s(b), c(b)]])
        Rz2 = np.array([[c(g), -s(g), 0], [s(g), c(g), 0], [0, 0, 1]])
        return Rz1 @ Rx @ Rz2
    else:
        raise ValueError(f"Unsupported Euler sequence: {seq}")


def _euler_from_R_zyx(R: np.ndarray) -> Tuple[bool, np.ndarray]:
    """Extract ZYX (yaw, pitch, roll) from rotation matrix; returns (ok, angles)."""
    # ZYX: R = Rz(yaw) Ry(pitch) Rx(roll)
    # pitch = asin(-R[2,0])
    sy = -float(R[2, 0])
    if abs(sy) >= 1.0:
        # Gimbal lock; roll and yaw coupled
        pitch = np.sign(sy) * (np.pi / 2)
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
        return False, np.array([yaw, pitch, roll])
    pitch = np.arcsin(sy)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return True, np.array([yaw, pitch, roll])


def _G_euler_zyx(angles: np.ndarray) -> np.ndarray:
    """
    Euler-rate map for ZYX: ω = G(φ) φ̇, φ=[yaw(Z), pitch(Y), roll(X)].
    We return G; caller can invert it for JA.
    """
    yaw, pitch, roll = angles
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    # One common parametrization:
    G = np.array(
        [
            [0.0, -sr, cr * cp],
            [0.0, cr, sr * cp],
            [1.0, 0.0, -sp],
        ],
        dtype=float,
    )
    return G


def _euler_from_R_zxz(R: np.ndarray) -> Tuple[bool, np.ndarray]:
    """Extract ZXZ angles; returns (ok, angles)."""
    # Guard singularities when sin(beta)≈0
    beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
    sb = np.sin(beta)
    if sb < 1e-9:
        alpha = 0.0
        gamma = np.arctan2(R[0, 1], R[0, 0])
        return False, np.array([alpha, beta, gamma])
    alpha = np.arctan2(R[0, 2], -R[1, 2])
    gamma = np.arctan2(R[2, 0], R[2, 1])
    return True, np.array([alpha, beta, gamma])


def _G_euler_zxz(angles: np.ndarray) -> np.ndarray:
    """
    Euler-rate map for ZXZ: ω = G(φ) φ̇, φ=[alpha(Z), beta(X), gamma(Z)].
    """
    alpha, beta, gamma = angles
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    sg, cg = np.sin(gamma), np.cos(gamma)
    G = np.array(
        [
            [0.0, sb, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, cb, 0.0],
        ],
        dtype=float,
    )
    return G


def _euler_rate_map_inverse_from_R(R: np.ndarray, seq: str) -> Tuple[bool, np.ndarray]:
    """
    Compute G^{-1}(φ) from rotation matrix for given Euler sequence.
    Returns (ok, Ginv). ok=False indicates near-singularity; caller may fallback.
    """
    seq = seq.upper()
    if seq == "ZYX":
        ok, ang = _euler_from_R_zyx(R)
        if not ok:
            return False, np.eye(3)
        G = _G_euler_zyx(ang)
        try:
            return True, np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return False, np.eye(3)
    elif seq == "ZXZ":
        ok, ang = _euler_from_R_zxz(R)
        if not ok:
            return False, np.eye(3)
        G = _G_euler_zxz(ang)
        try:
            return True, np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return False, np.eye(3)
    else:
        raise ValueError(f"Unsupported Euler sequence: {seq}")


# -------------------------- typing helper for .from_spec ----------------------

class RobotSpecLike:
    """Minimal protocol: any object with a `.data` mapping works (used in tests)."""
    data: Mapping[str, Any]
