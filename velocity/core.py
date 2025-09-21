# velocity/core.py
"""
Core kinematics for the Velocity Kinematics Toolkit.

Provides:
  - DHRobot (FK, geometric & analytic Jacobians)
  - URDFRobot placeholder
  - solvers: resolved_rates (DLS) and newton_ik (masked LS with log error)

Notes
-----
- Standard DH (Craig):  Tz(d) · Rz(θ) · Tx(a) · Rx(α).
- Jacobian columns: for revolute i, [ k_i × (p_e - p_i) ; k_i ], for prismatic i,
  [ k_i ; 0 ] — all expressed in the base frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .utils import (
    transl,
    trotz,
    trotx,
    mmul,
    pinv_damped,  # SVD-based damped pseudoinverse
)

# ------------------------------- Data classes ---------------------------------


@dataclass(slots=True)
class JointDH:
    name: str
    joint_type: str  # 'R' or 'P'
    alpha: float
    a: float
    d: float
    theta: float
    axis_local: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))

    def transform(self, q_i: float) -> np.ndarray:
        """
        {i-1}->i using Craig standard DH:
            T = Tz(d_i) · Rz(θ_i) · Tx(a_i) · Rx(α_i)
        θ_i = θ_offset + q_i (R), d_i = d_offset + q_i (P)
        """
        theta = self.theta + (q_i if self.joint_type.upper() == "R" else 0.0)
        di = self.d + (q_i if self.joint_type.upper() == "P" else 0.0)
        return mmul(transl(0, 0, di), trotz(theta), transl(self.a, 0, 0), trotx(self.alpha))


# ------------------------------ Robot abstractions ----------------------------


class _BaseRobot:
    def fk(self, q: np.ndarray) -> Dict[str, Any]:
        Ts, Tn = self._fk_all(q)
        return {"T_0e": Tn, "frames": Ts}

    def _fk_all(self, q: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def jacobian_geometric(self, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian_analytic(self, q: np.ndarray, euler: str = "ZYX") -> np.ndarray:
        raise NotImplementedError


class DHRobot(_BaseRobot):
    def __init__(self, joints: List[JointDH], tool: Optional[np.ndarray] = None, name: Optional[str] = None):
        self.joints: List[JointDH] = joints
        self.tool: np.ndarray = (np.eye(4) if tool is None else np.asarray(tool, dtype=float))
        self.name = name or "dh_robot"

    @staticmethod
    def from_spec(spec: "RobotSpecLike") -> "DHRobot":
        data = spec.data if hasattr(spec, "data") else spec
        joints: List[JointDH] = []
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

    def _fk_all(self, q: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        q = np.asarray(q, dtype=float).ravel()
        if q.size != len(self.joints):
            raise ValueError(f"q has length {q.size}, expected {len(self.joints)}")
        T = np.eye(4)
        frames: List[np.ndarray] = [T.copy()]  # base
        for ji, qi in zip(self.joints, q):
            T = T @ ji.transform(qi)
            frames.append(T.copy())
        Tn = T @ self.tool
        frames[-1] = Tn
        return frames, Tn

    def _axes_and_origins(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        axes: List[np.ndarray] = []
        origins: List[np.ndarray] = []
        for j, Ti in zip(self.joints, frames[:-1]):
            R = Ti[:3, :3]
            p = Ti[:3, 3]
            k = R @ j.axis_local
            axes.append(k)
            origins.append(p)
        return axes, origins

    def jacobian_geometric(self, q: np.ndarray) -> np.ndarray:
        """
        6×n geometric Jacobian via Jacobian-generating vectors (base frame):
            J_i = [ k_i × (p_e - p_i) ; k_i ]  (R)
            J_i = [ k_i               ; 0   ]  (P)
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

    def jacobian_analytic(self, q: np.ndarray, euler: str = "ZYX") -> np.ndarray:
        euler = euler.upper()
        Jg = self.jacobian_geometric(q)
        _, Tn = self._fk_all(np.asarray(q, dtype=float).ravel())
        R = Tn[:3, :3]
        ok, Ginv = _euler_rate_map_inverse_from_R(R, euler)
        if not ok:
            return Jg  # fallback near Euler-map singularities
        return np.vstack([Jg[:3, :], Ginv @ Jg[3:, :]])


class URDFRobot(_BaseRobot):
    def __init__(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "URDFRobot requires a URDF backend (e.g., urdfpy/pinocchio). "
            "Integrate your preferred library and implement from_spec(), "
            "_fk_all(), jacobian_geometric(), jacobian_analytic()."
        )

    @staticmethod
    def from_spec(spec: "RobotSpecLike") -> "URDFRobot":
        return URDFRobot()  # pragma: no cover


# ----------------------------------- Solvers ----------------------------------


class solvers:
    # ---------------------- Inverse Velocity (resolved rates) ----------------------
    @staticmethod
    def _task_mask_from_xdot(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Build a boolean mask of task rows to keep based on which desired
        components are numerically non-zero. If all are ~0, keep all rows.
        """
        mask = np.abs(x) > eps
        if not mask.any():
            mask[:] = True
        return mask

    @staticmethod
    def resolved_rates(
        J: np.ndarray,
        Xdot: np.ndarray,
        damping: Optional[float] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Solve q̇ from Ẋ = J q̇ using **masked** damped least squares.

        We only fit rows whose desired components are non-zero (|Ẋ_i|>eps). This
        prevents orientation (or other) rows with zero targets from polluting a
        translation-only task (the source of your failing test).

        Unweighted:
            q̇ = pinv_damped(J_sel, λ) Ẋ_sel
        Weighted (diagonal W on task space):
            let W^{1/2} = diag(sqrt(w_sel))
            q̇ = pinv_damped(W^{1/2} J_sel, λ) (W^{1/2} Ẋ_sel)
        """
        J = np.asarray(J, dtype=float)
        x = np.asarray(Xdot, dtype=float).reshape(-1)
        lam = float(damping) if damping is not None else 1e-9

        # Select only active task rows
        mask = solvers._task_mask_from_xdot(x)
        Jsel = J[mask, :]
        xsel = x[mask]

        if weights is None:
            return pinv_damped(Jsel, lam=lam) @ xsel
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.size != J.shape[0]:
                raise ValueError(f"weights has length {w.size}, expected {J.shape[0]}")
            wsel = w[mask]
            Wsqrt = np.diag(np.sqrt(wsel))
            Jw = Wsqrt @ Jsel
            Xw = Wsqrt @ xsel
            return pinv_damped(Jw, lam=lam) @ Xw

    # ------------------------- Orientation utility (log map) ----------------------
    @staticmethod
    def _rotvec_from_R(R: np.ndarray) -> np.ndarray:
        """
        Rotation-vector from rotation matrix via log map:
            R = exp([r]_x)  =>  r = vee(log R)
        Uses stable small-angle handling.
        """
        tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(tr)
        if theta < 1e-12:
            # first-order approximation
            return 0.5 * np.array([R[2, 1] - R[1, 2],
                                   R[0, 2] - R[2, 0],
                                   R[1, 0] - R[0, 1]])
        s = np.sin(theta)
        if abs(s) < 1e-12:
            # extremely close to pi: guarded scaling
            return 0.5 * np.array([R[2, 1] - R[1, 2],
                                   R[0, 2] - R[2, 0],
                                   R[1, 0] - R[0, 1]]) * (theta / (s if s != 0 else 1.0))
        K = theta / (2.0 * s)
        return K * np.array([R[2, 1] - R[1, 2],
                             R[0, 2] - R[2, 0],
                             R[1, 0] - R[0, 1]])

    # --------------------------- Newton–Raphson IK (masked) -----------------------
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

        Task selection (static mask built from x_target):
          - If 'p' in x_target, include translational rows (0..2).
          - If 'R' or 'euler' in x_target, include orientation rows (3..5).
          - If neither, defaults to position-only with current orientation held.
        """
        q = np.asarray(q0, dtype=float).ravel()
        p_des: Optional[np.ndarray] = None
        R_des: Optional[np.ndarray] = None

        pos_active = "p" in x_target
        if pos_active:
            p_des = np.asarray(x_target["p"], dtype=float).reshape(3)

        if "R" in x_target:
            R_des = np.asarray(x_target["R"], dtype=float).reshape(3, 3)
            ori_active = True
        elif "euler" in x_target:
            ed = x_target["euler"]
            seq = str(ed.get("seq", euler)).upper()
            ang = np.asarray(ed["angles"], dtype=float).reshape(3)
            R_des = _R_from_euler(seq, ang)
            ori_active = True
        else:
            ori_active = False

        if R_des is None:
            # Hold the initial orientation if none provided
            _, Tn0 = robot._fk_all(q)
            R_des = Tn0[:3, :3]

        # Static task mask
        task_mask = np.zeros(6, dtype=bool)
        if pos_active:
            task_mask[:3] = True
        if ori_active:
            task_mask[3:] = True
        if not task_mask.any():
            task_mask[:3] = True  # default: position-only

        res_hist: List[float] = []
        converged = False
        lam = 1e-6

        for k in range(1, max_iter + 1):
            _, Tn = robot._fk_all(q)
            R = Tn[:3, :3]
            p = Tn[:3, 3]

            # Residual
            dp = (p_des - p) if pos_active else np.zeros(3, dtype=float)
            if ori_active:
                # Orientation error as rotation vector of R_des * R^T
                Rerr = R_des @ R.T
                w_err = solvers._rotvec_from_R(Rerr)
            else:
                w_err = np.zeros(3, dtype=float)

            xerr_full = np.r_[dp, w_err]
            xerr = xerr_full[task_mask]

            res = float(np.linalg.norm(xerr))
            res_hist.append(res)
            if res < tol:
                converged = True
                break

            Jg = robot.jacobian_geometric(q)[task_mask, :]

            if weights is None:
                dq = pinv_damped(Jg, lam=lam) @ xerr
            else:
                w = np.asarray(weights, dtype=float).reshape(-1)
                if w.size != 6:
                    raise ValueError("weights must have length 6 for IK")
                wsel = w[task_mask]
                Wsqrt = np.diag(np.sqrt(wsel))
                dq = pinv_damped(Wsqrt @ Jg, lam=lam) @ (Wsqrt @ xerr)

            q = q + dq

        info = {"iters": k, "converged": converged, "res_norm": res_hist[-1], "res_hist": res_hist}
        return q, info


# --------------------------- Euler utilities ----------------------------------


def _R_from_euler(seq: str, ang: np.ndarray) -> np.ndarray:
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
    sy = -float(R[2, 0])
    if abs(sy) >= 1.0:
        pitch = np.sign(sy) * (np.pi / 2)
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
        return False, np.array([yaw, pitch, roll])
    pitch = np.arcsin(sy)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return True, np.array([yaw, pitch, roll])


def _G_euler_zyx(angles: np.ndarray) -> np.ndarray:
    yaw, pitch, roll = angles
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [0.0, -sr, cr * cp],
            [0.0,  cr, sr * cp],
            [1.0, 0.0,   -sp  ],
        ],
        dtype=float,
    )


def _euler_from_R_zxz(R: np.ndarray) -> Tuple[bool, np.ndarray]:
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
    alpha, beta, gamma = angles
    sb, cb = np.sin(beta), np.cos(beta)
    return np.array(
        [
            [0.0, sb, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, cb, 0.0],
        ],
        dtype=float,
    )


def _euler_rate_map_inverse_from_R(R: np.ndarray, seq: str) -> Tuple[bool, np.ndarray]:
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
    data: Mapping[str, Any]
