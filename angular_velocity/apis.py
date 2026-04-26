# angular_velocity/apis.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
from .core import Rotation, Transform, KinematicsEngine, Screw
from .io import IOManager

@dataclass
class AngularAPI:
    """High-level façade consumed by CLI or other apps."""
    # ❗️Use default_factory to avoid a mutable default in a dataclass field
    io: IOManager = field(default_factory=IOManager)

    # ---- Conversions & primitives ----
    def rotation_from_euler(self, order: str, angles: Tuple[float, float, float]) -> Rotation:
        return Rotation.from_euler(order, angles)

    def omega_from_Rdot(self, R: np.ndarray, Rdot: np.ndarray) -> np.ndarray:
        return Rotation(R).omega_tilde_from_Rdot(Rdot)

    def Rdot_from_omega(self, R: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return Rotation(R).Rdot_from_omega(omega)

    # ---- Velocity matrix & rigid-body velocity_kinematics ----
    def velocity_matrix(self, R: np.ndarray, d: np.ndarray, Rdot: np.ndarray, ddot: np.ndarray) -> np.ndarray:
        return Transform(R, d).velocity_matrix(Rdot, ddot)

    def rigid_point_velocity(self, omega_g: np.ndarray, r_gp: np.ndarray, d_gb: np.ndarray, d_gb_dot: np.ndarray) -> np.ndarray:
        return KinematicsEngine.rigid_body_point_velocity(omega_g, r_gp, d_gb, d_gb_dot)

    def screw_from_twist(self, twist6: np.ndarray) -> Screw:
        return Screw.from_twist(twist6)

    # ---- I/O helpers ----
    def persist_rotation(self, name: str, R: np.ndarray) -> None:
        self.io.save_npy(name, R)

    def load_rotation(self, name: str) -> np.ndarray:
        return self.io.load_npy(name)
