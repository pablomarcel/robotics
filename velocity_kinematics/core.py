# velocity_kinematics/core.py
"""
Core kinematics for the Velocity Kinematics Toolkit.

Provides:
  - DHRobot (FK, geometric & analytic Jacobians)
  - URDFRobot (FK, geometric & analytic Jacobians via urdfpy)
  - solvers: resolved_rates (masked DLS) and newton_ik (masked LS with log error)

Notes
-----
- Standard DH (Craig):  Tz(d) · Rz(θ) · Tx(a) · Rx(α).
- Jacobian columns: for revolute i, [ k_i × (p_e - p_i) ; k_i ], for prismatic i,
  [ k_i ; 0 ] — all expressed in the base frame.

URDF assumptions
----------------
- We support tree URDFs; for Jacobians we use the unique parent chain from base
  link to the chosen end-effector link. Fixed joints along the chain are applied_dynamics
  to FK (no dof). Actuated joints are those with type in {'revolute','continuous','prismatic'}.
- Joint axis is defined in the joint frame (URDF spec). We propagate the joint
  frame pose along the chain at the CURRENT configuration and rotate axes into
  the base/world frame for the Jacobian columns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
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


# ---------------------------------- URDFRobot ---------------------------------

def _urdfpy_load(path: str):
    """
    Load a URDF file via urdfpy across versions.

    urdfpy >= 0.0.18  : URDF.load(path_planning)
    older urdfpy      : URDF.from_xml_file(path_planning)
    """
    try:
        from urdfpy import URDF  # type: ignore
    except Exception as e:
        raise RuntimeError("URDF support requires `urdfpy`. Install with: pip install urdfpy") from e

    if hasattr(URDF, "load"):
        return URDF.load(path)
    if hasattr(URDF, "from_xml_file"):
        return URDF.from_xml_file(path)
    raise RuntimeError("Unsupported urdfpy version: no URDF.load or URDF.from_xml_file")


def _rot_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation_kinematics for unit axis (3,) and angle (rad) -> (3,3)."""
    a = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(a)
    if n == 0.0:
        return np.eye(3)
    a = a / n
    x, y, z = a
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def _hom(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Assemble 4x4 from (3,3) and (3,)."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p.reshape(3)
    return T


class URDFRobot(_BaseRobot):
    """
    URDF-backed robot_dynamics using `urdfpy`.

    - FK computed along the base→EE chain using joint origins and motion_kinematics
    - Geometric Jacobian computed analytically from world joint axes/origins
    - Analytic Jacobian via Euler maps (ZYX/ZXZ)

    Construction (preferred): URDFRobot.from_spec(spec)
    spec can be:
      - path_planning string/Path to .urdf
      - dict with keys:
          urdf: path_planning to .urdf
          base_link: optional base link name (defaults to robot_dynamics.base_link)
          ee_link:   optional end-effector link (defaults to a leaf)
          name:      optional robot_dynamics name
    """

    def __init__(
        self,
        urdf_obj: "URDF",
        *,
        base_link: Optional[str] = None,
        ee_link: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self._urdf = urdf_obj
        self.name = name or (getattr(urdf_obj, "name", None) or "urdf_robot")

        # base_link may be a Link object or a string depending on urdfpy version.
        bl = getattr(urdf_obj, "base_link", None)
        if hasattr(bl, "name"):
            bl = bl.name
        self._base_link = base_link or bl or urdf_obj.links[0].name

        self._ee_link = ee_link or self._default_leaf_link()

        # Build parent map child_link -> (parent_link, joint)
        self._parent_of: Dict[str, Tuple[str, Any]] = {}
        for j in urdf_obj.joints:
            parent = getattr(j, "parent", None)
            child = getattr(j, "child", None)
            # urdfpy typically uses strings; be robust to Link objects
            if hasattr(parent, "name"):
                parent = parent.name
            if hasattr(child, "name"):
                child = child.name
            if parent and child:
                self._parent_of[child] = (parent, j)

        # Build ordered chain (parent_link, joint, child_link) ... to ee
        self._chain = self._build_chain(self._base_link, self._ee_link)
        # Extract actuated joints in order (exclude fixed)
        self._actuated: List[Any] = [
            j for (_, j, _) in self._chain if j.joint_type in ("revolute", "continuous", "prismatic")
        ]
        self._actuated_names: List[str] = [j.name for j in self._actuated]

    # ---------- construction helpers ----------

    @staticmethod
    def from_spec(spec: "RobotSpecLike") -> "URDFRobot":
        data = spec.data if hasattr(spec, "data") else spec
        if isinstance(data, (str, os.PathLike, bytes)):
            path = str(data)
            rob = _urdfpy_load(path)
            return URDFRobot(rob)
        elif isinstance(data, dict):
            path = data.get("urdf") or data.get("path_planning") or data.get("file")
            if not path:
                raise ValueError("URDF spec dict must include key 'urdf' with a path_planning to the .urdf file.")
            rob = _urdfpy_load(str(path))
            return URDFRobot(
                rob,
                base_link=data.get("base_link"),
                ee_link=data.get("ee_link"),
                name=data.get("name"),
            )
        else:
            raise ValueError("Unsupported spec for URDFRobot.from_spec")

    def _default_leaf_link(self) -> str:
        link_names = {getattr(lk, "name", str(lk)) for lk in self._urdf.links}
        parents = {
            (j.parent.name if hasattr(j.parent, "name") else j.parent)
            for j in self._urdf.joints
            if getattr(j, "parent", None)
        }
        leaves = sorted(list(link_names - parents))
        if leaves:
            return leaves[-1]
        # Fallback
        last = self._urdf.links[-1]
        return last.name if hasattr(last, "name") else str(last)

    def _build_chain(self, base_link: str, tip_link: str) -> List[Tuple[str, Any, str]]:
        """Return ordered list of (parent_link, joint, child_link) from base→tip."""
        chain_rev: List[Tuple[str, Any, str]] = []
        cur = tip_link
        while cur != base_link:
            if cur not in self._parent_of:
                raise ValueError(f"No path_planning from base link '{base_link}' to tip link '{tip_link}'")
            parent, joint = self._parent_of[cur]
            chain_rev.append((parent, joint, cur))
            cur = parent
        return list(reversed(chain_rev))

    # ---------- FK internals ----------

    def _cfg_from_q(self, q: np.ndarray) -> Dict[str, float]:
        q = np.asarray(q, dtype=float).ravel()
        if q.size != len(self._actuated):
            raise ValueError(f"q has length {q.size}, expected {len(self._actuated)}")
        return {j.name: float(val) for j, val in zip(self._actuated, q)}

    @staticmethod
    def _origin_matrix(joint: Any) -> np.ndarray:
        """Convert urdfpy joint.origin (xyz/rpy) to 4x4."""
        T = getattr(joint, "origin", None)
        if T is None:
            return np.eye(4, dtype=float)
        T = np.asarray(T, dtype=float)
        if T.shape == (4, 4):
            return T
        # Fallback: try fields xyz, rpy (rare in urdfpy)
        xyz = np.asarray(getattr(joint, "origin_xyz", [0, 0, 0]), dtype=float).reshape(3)
        rpy = np.asarray(getattr(joint, "origin_rpy", [0, 0, 0]), dtype=float).reshape(3)
        # Build best-effort 4x4 (Rz*Ry*Rx if you wish to refine later)
        Rx = np.array([[1, 0, 0], [0, np.cos(rpy[0]), -np.sin(rpy[0])], [0, np.sin(rpy[0]), np.cos(rpy[0])]])
        Ry = np.array([[np.cos(rpy[1]), 0, np.sin(rpy[1])], [0, 1, 0], [-np.sin(rpy[1]), 0, np.cos(rpy[1])]])
        Rz = np.array([[np.cos(rpy[2]), -np.sin(rpy[2]), 0], [np.sin(rpy[2]), np.cos(rpy[2]), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        T4 = np.eye(4)
        T4[:3, :3] = R
        T4[:3, 3] = xyz
        return T4

    @staticmethod
    def _joint_motion(joint: Any, q_i: float) -> np.ndarray:
        """Return 4x4 transform for this joint at position q_i in its local joint frame."""
        jtype = joint.joint_type
        axis = np.asarray(getattr(joint, "axis", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0.0:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            axis = axis / axis_norm
        if jtype in ("revolute", "continuous"):
            R = _rot_from_axis_angle(axis, q_i)
            return _hom(R, np.zeros(3))
        elif jtype == "prismatic":
            return _hom(np.eye(3), axis * q_i)
        elif jtype == "fixed" or jtype is None:
            return np.eye(4)
        else:
            raise ValueError(f"Unsupported URDF joint type: {jtype}")

    def _fk_all(self, q: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute FK along the base→EE chain. For consistency with DHRobot,
        return frames=[I, ..., T_0e].
        """
        q = np.asarray(q, dtype=float).ravel()
        qmap = self._cfg_from_q(q) if len(self._actuated) else {}
        T = np.eye(4, dtype=float)
        frames: List[np.ndarray] = [T.copy()]
        for parent, joint, child in self._chain:
            T = T @ self._origin_matrix(joint)  # parent -> joint frame (fixed)
            if joint.joint_type in ("revolute", "continuous", "prismatic"):
                q_i = qmap[joint.name]
                T = T @ self._joint_motion(joint, q_i)  # joint motion_kinematics to child link
            frames.append(T.copy())
        T_0e = frames[-1]
        return frames, T_0e

    # ---------- Jacobians ----------

    def jacobian_geometric(self, q: np.ndarray) -> np.ndarray:
        """
        6×n geometric Jacobian for base→EE chain.

        For each actuated joint i:
          - k_i (world) = R_world @ axis_i (joint axis in joint frame)
          - p_i (world) = position of the joint origin (after fixed transforms and
            preceding joints). We use the pose just BEFORE applying this joint's
            own motion_kinematics (axis orientation_kinematics is invariant to its own rotation_kinematics).
          - J_i = [ k_i × (p_e - p_i) ; k_i ] for revolute/continuous
                  [ k_i               ; 0   ] for prismatic
        """
        q = np.asarray(q, dtype=float).ravel()
        if q.size != len(self._actuated):
            raise ValueError(f"q has length {q.size}, expected {len(self._actuated)}")

        # First pass to get p_e
        _, T_e = self._fk_all(q)
        p_e = T_e[:3, 3]

        # Walk chain while tracking world pose up to each joint origin
        J_cols: List[np.ndarray] = []
        T_world = np.eye(4, dtype=float)
        q_iter = iter(q)
        for parent, joint, child in self._chain:
            # World pose at this joint origin (before motion_kinematics)
            T_world = T_world @ self._origin_matrix(joint)
            p_i = T_world[:3, 3]
            R_w = T_world[:3, :3]
            jtype = joint.joint_type

            if jtype in ("revolute", "continuous", "prismatic"):
                axis_local = np.asarray(getattr(joint, "axis", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
                axn = np.linalg.norm(axis_local)
                if axn == 0.0:
                    axis_local = np.array([0.0, 0.0, 1.0], dtype=float)
                    axn = 1.0
                k_i = R_w @ (axis_local / axn)

                if jtype in ("revolute", "continuous"):
                    Jcol = np.zeros(6, dtype=float)
                    Jcol[:3] = np.cross(k_i, p_e - p_i)
                    Jcol[3:] = k_i
                else:  # prismatic
                    Jcol = np.zeros(6, dtype=float)
                    Jcol[:3] = k_i
                J_cols.append(Jcol)

                # Advance through the joint motion_kinematics using the provided q
                q_i = next(q_iter)
                T_world = T_world @ self._joint_motion(joint, q_i)
            else:
                # Fixed joint: no column, just advance
                T_world = T_world  # already included origin

        if len(J_cols) != len(self._actuated):
            raise RuntimeError("Jacobian construction error: column count mismatch.")

        return np.stack(J_cols, axis=1)

    def jacobian_analytic(self, q: np.ndarray, euler: str = "ZYX") -> np.ndarray:
        euler = euler.upper()
        Jg = self.jacobian_geometric(q)
        _, Tn = self._fk_all(np.asarray(q, dtype=float).ravel())
        R = Tn[:3, :3]
        ok, Ginv = _euler_rate_map_inverse_from_R(R, euler)
        if not ok:
            return Jg  # graceful fallback near singular Euler maps
        return np.vstack([Jg[:3, :], Ginv @ Jg[3:, :]])


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

        We only fit rows whose desired components are non-zero (|Ẋ_i|>eps).
        This prevents orientation_kinematics (or other) rows with zero targets from
        polluting a translation-only task.

        Unweighted:
            q̇ = pinv_damped(J_sel, λ) Ẋ_sel
        Weighted (diagonal W on task space):
            let W^{1/2} = diag(sqrt(w_sel))
            q̇ = pinv_damped(W^{1/2} J_sel, λ) (W^{1/2} Ẋ_sel)
        """
        J = np.asarray(J, dtype=float)
        x = np.asarray(Xdot, dtype=float).reshape(-1)
        lam = float(damping) if damping is not None else 1e-9

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
        Rotation-vector from rotation_kinematics matrix via log map:
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
        Newton–Raphson inverse_kinematics kinematics on pose x = [p; orientation_kinematics].

        Task selection (static mask built from x_target):
          - If 'p' in x_target, include translational rows (0..2).
          - If 'R' or 'euler' in x_target, include orientation_kinematics rows (3..5).
          - If neither, defaults to position-only with current orientation_kinematics held.
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
            # Hold the initial orientation_kinematics if none provided
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
                # Orientation error as rotation_kinematics vector of R_des * R^T
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
