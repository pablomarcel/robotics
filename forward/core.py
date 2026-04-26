# forward/core.py
"""
Core math & object model for forward kinematics (FK) and Jacobians.

This module is intentionally dependency-light (NumPy + our own utils) so it
is easy to unit-test and reason about. It supports three link styles:

- :class:`DHLink`   — Standard Denavit–Hartenberg (Eq. 5.5 style: Tz(d) Rz(θ) Tx(a) Rx(α))
- :class:`MDHLink`  — Modified DH (Craig)       (Rz(θ) Tz(d) Tx(a) Rx(α))
- :class:`PoELink`  — Product-of-Exponentials with 6D screw axes in **space frame**

A :class:`SerialChain` composes links and computes:

- Forward kinematics:  ^0T_n(q)
- Analytical **space Jacobian** J_s(q)
- Analytical **body Jacobian**  J_b(q)  (via adjoint transform)

Conventions
-----------
- R^3 vectors are column vectors when embedded in 4x4 matrices.
- Plücker twist stacking order for Jacobians is ``[ω; v]`` (6×1).
- For DH/MDH links, the joint axis is the local +z axis; we express it in the
  base frame via accumulated rotations.
- For PoE chains, link screws are provided in the **space** frame; we use
  Adjoint products per Lynch & Park (modern robotics).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .utils import timed, skew, adjoint


# --------------------------------------------------------------------------
# Rotations and Transforms
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Rotation:
    """SO(3) helpers."""

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    @staticmethod
    def axis_angle(R: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute axis–angle (φ, u) from a rotation_kinematics matrix.

        Returns
        -------
        (phi, u) with phi in radians and u as a 3-vector (unit).
        """
        tr = np.trace(R)
        phi = np.arccos(np.clip(0.5 * (tr - 1.0), -1.0, 1.0))
        if np.isclose(phi, 0.0):
            return 0.0, np.array([1.0, 0.0, 0.0])
        u = (1.0 / (2.0 * np.sin(phi))) * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
        n = np.linalg.norm(u)
        return phi, u / (n if n > 0 else 1.0)


@dataclass
class Transform:
    """
    SE(3) homogeneous transform T = [[R, t], [0, 1]].

    Attributes
    ----------
    R : (3,3) ndarray
        Rotation block.
    t : (3,) ndarray
        Translation vector.
    """

    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def eye() -> "Transform":
        return Transform(np.eye(3), np.zeros(3))

    @staticmethod
    def from_matrix(T: np.ndarray) -> "Transform":
        return Transform(T[:3, :3].copy(), T[:3, 3].copy())

    @staticmethod
    def from_Rt(R: np.ndarray, t: np.ndarray) -> "Transform":
        return Transform(R.copy(), t.copy())

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def inv(self) -> "Transform":
        RT = self.R.T
        return Transform(RT, -RT @ self.t)

    def apply(self, p: np.ndarray) -> np.ndarray:
        """
        Apply transform to point(s).

        Parameters
        ----------
        p : (3,) or (3,n)

        Returns
        -------
        (3,) or (3,n)
        """
        if p.ndim == 1:
            return self.R @ p + self.t
        return self.R @ p + self.t.reshape(3, 1)

    def __matmul__(self, other: "Transform") -> "Transform":
        T = self.as_matrix() @ other.as_matrix()
        return Transform.from_matrix(T)


# --------------------------------------------------------------------------
# Links
# --------------------------------------------------------------------------

class Link:
    """Abstract base class for kinematic links."""

    joint_type: str = "R"  # 'R' or 'P'

    def fk(self, q: float) -> Transform:
        """Return ^{i-1}T_i(q)."""
        raise NotImplementedError


class DHLink(Link):
    """
    Standard DH link (Craig Eq. 3.6 form): Tz(d) · Rz(θ) · Tx(a) · Rx(α).

    Parameters
    ----------
    a : float
        Link length.
    alpha : float
        Link twist (radians).
    d : float
        Link offset (for R joint it's constant; for P joint it's variable).
    theta_offset : float
        Constant offset added to revolute joint angle.
    joint_type : {'R','P'}
        Revolute or prismatic.
    """

    def __init__(
        self,
        a: float,
        alpha: float,
        d: float,
        theta_offset: float = 0.0,
        joint_type: str = "R",
    ):
        self.a = float(a)
        self.alpha = float(alpha)
        self.d = float(d)
        self.theta_offset = float(theta_offset)
        self.joint_type = joint_type.upper()

    def A(self, q: float) -> np.ndarray:
        if self.joint_type == "R":
            theta, d = q + self.theta_offset, self.d
        else:
            theta, d = self.theta_offset, self.d + q
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array(
            [
                [ct, -st * ca, st * sa, self.a * ct],
                [st, ct * ca, -ct * sa, self.a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    def fk(self, q: float) -> Transform:
        return Transform.from_matrix(self.A(q))


class MDHLink(Link):
    """
    Modified DH link: Rz(θ) · Tz(d) · Tx(a) · Rx(α).

    Same constructor as :class:`DHLink`.
    """

    def __init__(
        self,
        a: float,
        alpha: float,
        d: float,
        theta_offset: float = 0.0,
        joint_type: str = "R",
    ):
        self.a = float(a)
        self.alpha = float(alpha)
        self.d = float(d)
        self.theta_offset = float(theta_offset)
        self.joint_type = joint_type.upper()

    def A(self, q: float) -> np.ndarray:
        if self.joint_type == "R":
            theta, d = q + self.theta_offset, self.d
        else:
            theta, d = self.theta_offset, self.d + q
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        ct, st = np.cos(theta), np.sin(theta)
        # Rz(θ) Tz(d) Tx(a) Rx(α) → same closed form as DH because of order properties
        return np.array(
            [
                [ct, -st * ca, st * sa, self.a * ct],
                [st, ct * ca, -ct * sa, self.a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    def fk(self, q: float) -> Transform:
        return Transform.from_matrix(self.A(q))


class PoELink(Link):
    """
    Product-of-Exponentials link with **space-frame screw axis**.

    Parameters
    ----------
    omega : (3,) array
        Angular axis of the screw; zero vector for prismatic.
    v : (3,) array
        Linear part of the screw.
    """

    def __init__(self, omega: np.ndarray, v: np.ndarray):
        w = np.asarray(omega, dtype=float).reshape(3)
        vc = np.asarray(v, dtype=float).reshape(3)
        self.omega = w
        self.v = vc
        self.joint_type = "R" if np.linalg.norm(w) > 1e-12 else "P"

    def exp(self, q: float) -> np.ndarray:
        """
        Matrix exponential exp([ξ]^ q) in SE(3).
        """
        w = self.omega
        v = self.v
        wn = np.linalg.norm(w)
        T = np.eye(4)
        if wn < 1e-12:  # prismatic
            T[:3, :3] = np.eye(3)
            T[:3, 3] = v * q
            return T
        wh = skew(w)
        th = q
        R = np.eye(3) + np.sin(th) * wh + (1 - np.cos(th)) * (wh @ wh)
        G = (
            np.eye(3) * th
            + (1 - np.cos(th)) * wh
            + (th - np.sin(th)) * (wh @ wh)
        )
        p = G @ v
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def fk(self, q: float) -> Transform:
        return Transform.from_matrix(self.exp(q))

    def screw6(self) -> np.ndarray:
        """Return 6×1 screw [ω; v] (in space frame)."""
        return np.concatenate([self.omega, self.v], axis=0).reshape(6,)


# --------------------------------------------------------------------------
# Serial chain
# --------------------------------------------------------------------------

class SerialChain:
    """
    Serial manipulator composed of :class:`Link` instances.

    Parameters
    ----------
    links : Sequence[Link]
        Ordered links from base (0) to tip (n-1).
    M : (4,4) ndarray, optional
        Fixed tool transform applied at the end (home/tool offset).
    name : str
        Identifier.

    Notes
    -----
    - FK is computed as ``T = A1(q1) · A2(q2) · ... · An(qn) · M``.
    - For PoE, ``Ai(qi)`` is the matrix exponential ``exp([Si]^ qi)`` and ``M``
      should be the home pose of the end-effector.
    """

    def __init__(self, links: Sequence[Link], M: Optional[np.ndarray] = None, name: str = "robot"):
        self._links: List[Link] = list(links)
        self.M = np.eye(4) if M is None else np.array(M, dtype=float)
        self.name = name

    # ---------------------------- basic getters ----------------------------

    def n(self) -> int:
        return len(self._links)

    def links(self) -> List[Link]:
        return list(self._links)

    # ---------------------------- forward kinematics -----------------------

    @timed
    def fkine(self, q: Sequence[float]) -> Transform:
        """
        Compute forward kinematics ^0T_n(q).
        """
        qv = np.asarray(q, dtype=float).reshape(-1)
        if qv.size != self.n():
            raise ValueError(f"q has length {qv.size}, expected {self.n()}")
        T = np.eye(4)
        for link, qi in zip(self._links, qv):
            T = T @ link.fk(qi).as_matrix()
        T = T @ self.M
        return Transform.from_matrix(T)

    # ---------------------------- Jacobians --------------------------------

    def jacobian_space(self, q: Sequence[float]) -> np.ndarray:
        """
        Analytical **space** Jacobian J_s(q) (6×n).

        - For DH/MDH: computed from accumulated frame origins/orientations.
          Column i uses z_{i-1} (axis in space) and p_{i-1}.
            * Revolute : [ ω_i = z ; v_i = p × z ]
            * Prismatic: [ ω_i = 0 ; v_i = z ]
        - For PoE   : standard adjoint-based product-of-exponentials Jacobian.
        """
        qv = np.asarray(q, dtype=float).reshape(-1)
        if qv.size != self.n():
            raise ValueError(f"q has length {qv.size}, expected {self.n()}")

        # PoE path: we have screws in the space frame.
        if all(isinstance(L, PoELink) for L in self._links):
            n = self.n()
            J = np.zeros((6, n))
            T_prev = np.eye(4)
            for i, (L, qi) in enumerate(zip(self._links, qv), start=1):
                S = L.screw6().reshape(6, 1)
                if i == 1:
                    J[:, 0:1] = S
                else:
                    Ad = adjoint(T_prev)
                    J[:, i - 1 : i] = Ad @ S
                T_prev = T_prev @ L.fk(qi).as_matrix()
            return J

        # DH/MDH path
        # Accumulate ^0T_i for all i (including ^0T_0 = I)
        Ts: List[np.ndarray] = [np.eye(4)]
        T = np.eye(4)
        for L, qi in zip(self._links, qv):
            T = T @ L.fk(qi).as_matrix()
            Ts.append(T.copy())  # ^0T_i

        n = self.n()
        J = np.zeros((6, n))
        z = np.array([0.0, 0.0, 1.0])

        for i in range(n):
            T_im1 = Ts[i]              # ^0T_{i-1}
            R_im1 = T_im1[:3, :3]
            p_im1 = T_im1[:3, 3]
            z_axis = R_im1 @ z

            joint = self._links[i]
            if joint.joint_type == "R":
                omega = z_axis
                v = np.cross(p_im1, z_axis)
            else:
                omega = np.zeros(3)
                v = z_axis
            J[:, i] = np.concatenate([omega, v])
        return J

    def jacobian_body(self, q: Sequence[float]) -> np.ndarray:
        """
        Analytical **body** Jacobian J_b(q) (6×n), related by
            J_b(q) = Ad_{T(q)}^{-1} · J_s(q)
        where T(q) = ^0T_n(q).
        """
        T = self.fkine(q).as_matrix()
        Ad_inv = np.linalg.inv(adjoint(T))
        return Ad_inv @ self.jacobian_space(q)
