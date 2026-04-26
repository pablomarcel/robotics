"""
Angular: an OOP toolkit for angular_velocity velocity_kinematics & derivative kinematics.

Covers Eq. 7.1–7.416 foundations:
- SO(3)/SE(3) primitives (skew/vee, Rodrigues, exp/log small twist)
- ω-tilde relations, frame transforms, relative/additive ω
- Transport theorem & mixed derivatives
- SE(3) velocity_kinematics matrix (dot(T) T^{-1})
- Rigid-body velocity_kinematics with moving origin
- Screw axis & pitch; instantaneous centers (planar)
- DH links w/ revolute/prismatic velocity_kinematics coefficients

Public API is re-exported from .core and .apis
"""
from .core import Rotation, Transform, AngularVelocity, Screw, Frame, KinematicsEngine
from .apis import AngularAPI
__all__ = [
    "Rotation", "Transform", "AngularVelocity", "Screw", "Frame",
    "KinematicsEngine", "AngularAPI"
]
