from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from .utils import skew, vee, ensure_rotation, block44, adjoint, motion_cross

Array = np.ndarray

@dataclass
class Rotation:
    """SO(3) rotation with basic differential kinematics."""
    R: Array = field(repr=False)

    def __post_init__(self):
        from .utils import is_rotation_matrix
        if not is_rotation_matrix(self.R):
            raise ValueError("R must be a valid rotation matrix.")

    @classmethod
    def from_axis_angle(cls, u: Array, phi: float) -> "Rotation":
        u = np.asarray(u, dtype=float).reshape(3); u = u / np.linalg.norm(u)
        K = skew(u); I = np.eye(3)
        R = I + np.sin(phi)*K + (1-np.cos(phi))*(K@K)  # Rodrigues
        return cls(R)

    @classmethod
    def from_euler(cls, order: str, angles: Tuple[float,float,float]) -> "Rotation":
        """Create from intrinsic XYZ-like order. Minimal, works for standard 'ZYX','XYZ', etc."""
        def Rx(a): return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        def Ry(a): return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        maps = {'X':Rx, 'Y':Ry, 'Z':Rz}
        R = np.eye(3)
        for ax, ang in zip(order.upper(), angles):
            R = R @ maps[ax](ang)
        return cls(R)

    @ensure_rotation
    def omega_tilde_from_Rdot(self, Rdot: Array) -> Array:
        """Eq. (7.21): ω̃ = Ṙ Rᵀ."""
        return Rdot @ self.R.T

    @ensure_rotation
    def Rdot_from_omega(self, omega: Array) -> Array:
        """Eq. (7.22): Ṙ = ω̃ R."""
        return skew(omega) @ self.R

    @ensure_rotation
    def change_frame(self, R_to: Array) -> "Rotation":
        """Transform this rotation into a new basis: R_new = R_to * R."""
        return Rotation(R_to @ self.R)

@dataclass
class AngularVelocity:
    """Angular velocity in both vector and matrix forms with frame transforms."""
    omega: Array  # shape (3,)
    frame: str = "G"

    @property
    def tilde(self) -> Array:
        return skew(self.omega)

    def in_frame(self, R_to_from: Array) -> "AngularVelocity":
        """ω^to = R_to_from * ω^from"""
        return AngularVelocity(R_to_from @ self.omega, frame=self.frame)

@dataclass
class Transform:
    """SE(3) transform with velocity matrix support."""
    R: Array
    d: Array

    def as_matrix(self) -> Array:
        return block44(self.R, self.d)

    def velocity_matrix(self, Rdot: Array, ddot: Array) -> Array:
        """Eq. (7.351): V = Ṫ T^{-1} = [[ω̃, v],[0,0]] where v = ḋ - ω̃ d."""
        Omega = Rdot @ self.R.T
        v = ddot.reshape(3) - Omega @ self.d.reshape(3)
        V = np.zeros((4,4))
        V[:3,:3] = Omega
        V[:3,3]  = v
        return V

    def adjoint(self) -> Array:
        return adjoint(self.as_matrix())

@dataclass
class Screw:
    """Screw axis and pitch."""
    s: Array  # unit direction (3,)
    m: Array  # moment (3,) so that line: r × s + m = 0
    pitch: float  # p

    @classmethod
    def from_twist(cls, V: Array) -> "Screw":
        """Given spatial twist [ω; v], recover axis s, moment m, and pitch p (planar/3D)."""
        w = V[:3]; v = V[3:]
        wnorm2 = float(np.dot(w,w))
        if wnorm2 < 1e-12:  # pure translation → infinite pitch; pick s along v
            s = v / (np.linalg.norm(v)+1e-12); p = np.inf; m = np.zeros(3)
            return cls(s=s, m=m, pitch=p)
        s = w / np.sqrt(wnorm2)
        p = float(np.dot(s, v) / wnorm2)  # p = (s·v)/||w||^2
        m = np.cross(v, s) / wnorm2
        return cls(s=s, m=m, pitch=p)

@dataclass
class Frame:
    """A coordinate frame label with fixed transform to a parent."""
    name: str
    parent: Optional["Frame"] = None
    T_parent_this: Optional[Transform] = None

class KinematicsEngine:
    """
    High-level façade mapping textbook identities to code.
    Designed for testing (pure functions + deterministic behavior).
    """

    # ---- Eq. 7.26–7.29: v = ω × r  and frame transforms ----
    @staticmethod
    def point_velocity_global(R: Array, omega_g: Array, r_g: Array) -> Array:
        return np.cross(omega_g.reshape(3), r_g.reshape(3))

    @staticmethod
    def point_velocity_body(R: Array, omega_b: Array, r_b: Array) -> Array:
        return np.cross(omega_b.reshape(3), r_b.reshape(3))

    # ---- Eq. 7.214 (derivative transform / transport theorem) ----
    @staticmethod
    def transport_derivative(B_vec: Array, omega_b: Array, B_vec_dot_simple: Array) -> Array:
        """{}^G d/dt ({}^B v) expressed in B = {}^B d/dt v  +  ω^B × v"""
        return B_vec_dot_simple.reshape(3) + np.cross(omega_b.reshape(3), B_vec.reshape(3))

    # ---- Eq. 7.264–7.269: rigid body velocity with moving origin ----
    @staticmethod
    def rigid_body_point_velocity(omega_g: Array, r_gp: Array, d_gb: Array, d_gb_dot: Array) -> Array:
        return np.cross(omega_g.reshape(3), (r_gp - d_gb).reshape(3)) + d_gb_dot.reshape(3)

    # ---- Eq. 7.351 etc: velocity matrix and inverse mapping ----
    @staticmethod
    def velocity_matrix(R: Array, d: Array, Rdot: Array, ddot: Array) -> Array:
        return Transform(R, d).velocity_matrix(Rdot, ddot)

    # ---- Eq. 7.367/7.373: differential transform from small twist ----
    @staticmethod
    def differential_transform(dphiu: Array, dd: Array) -> Array:
        dR = np.eye(3) + skew(dphiu.reshape(3))
        T = np.eye(4); T[:3,:3] = dR; T[:3,3] = dd.reshape(3)
        return T - np.eye(4)  # ≈ [I + dD][I + dR] - I

    # ---- Eq. 7.281/7.283: instantaneous center (planar) ----
    @staticmethod
    def planar_instantaneous_center(d_gb: Array, d_gb_dot: Array, omega_z: float) -> Array:
        """Return r_G0(t) in 2D: r0 = d_B - (1/ω) ḋ_B (Eq. 7.283)."""
        if abs(omega_z) < 1e-12:
            raise ZeroDivisionError("ω must be nonzero for planar instantaneous center.")
        return d_gb.reshape(3) - (1.0/omega_z) * d_gb_dot.reshape(3)

    # ---- DH velocity coefficient matrices (7.361–7.363) ----
    @staticmethod
    def delta_revolute() -> Array:
        return np.array([[0, -1, 0, 0],
                         [1,  0, 0, 0],
                         [0,  0, 0, 0],
                         [0,  0, 0, 0]], float)

    @staticmethod
    def delta_prismatic() -> Array:
        return np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]], float)
