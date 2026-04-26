# forward/design.py
"""
Preset builders for common manipulators (SCARA, spherical wrists, planar arms).

All presets are expressed with **standard DH** links so they remain compatible
with the rest of the toolkit (FK, space/body Jacobians). Tool offsets are
applied as the fixed terminal transform ``M`` in :class:`forward.core.SerialChain`.

Notes on spherical wrists
-------------------------
A spherical wrist is a 3-DOF orientation_kinematics mechanism with three intersecting,
orthogonal joint axes. Although the *rotation_kinematics orders* are described in terms
of rotations about global X/Y/Z (e.g., Z–X–Z), DH parameterizations still use
local **z** joint axes; we achieve the desired rotation_kinematics orders by selecting
link twists (α) that realign each successive local z-axis with the intended
global axis before applying that joint's θ.

The presets below use the following α patterns (all links have a=0, d=0):

- Type 1 (Roll–Pitch–Roll ≈ Z–X–Z)
    α = [-π/2, +π/2, 0]
- Type 2 (Roll–Pitch–Yaw ≈ Z–Y–Z, “Euler wrist” variant)
    α = [+π/2, -π/2, 0]
- Type 3 (Pitch–Yaw–Roll ≈ X–Y–Z)
    α = [-π/2, -π/2, 0]

These produce the intended effective rotation_kinematics sequences while keeping the DH
convention that each joint actuates around its local z-axis.

If you need a different convention (MDH or PoE), you can add parallel presets
or feed equivalent screws into the PoE path elsewhere in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .core import DHLink, SerialChain, Transform, Rotation


# ------------------------------ Utilities --------------------------------- #

def _tool_Tz(d: float) -> np.ndarray:
    """Return a 4×4 homogeneous transform representing a pure translation Tz(d)."""
    M = np.eye(4)
    M[2, 3] = float(d)
    return M


# ------------------------------- Planar 2R -------------------------------- #

def planar_2r(l1: float, l2: float, *, name: str = "planar_2R") -> SerialChain:
    """
    Build a simple planar 2R arm in the x–y plane.

    Standard DH arrangement:
        - Joint 1: revolute about z0 at base
        - Joint 2: revolute about z1 at elbow
        - a2 carries link length l1; the fixed tool offset M carries l2

    Parameters
    ----------
    l1, l2 : float
        Link lengths.

    Returns
    -------
    SerialChain
    """
    L1 = DHLink(a=0.0, alpha=0.0, d=0.0, joint_type="R")
    L2 = DHLink(a=l1, alpha=0.0, d=0.0, joint_type="R")
    M = Transform.from_Rt(np.eye(3), np.array([l2, 0.0, 0.0])).as_matrix()
    return SerialChain([L1, L2], M=M, name=name)


# -------------------------------- SCARA ----------------------------------- #

def scara(l1: float, l2: float, d_home: float = 0.0, *, wrist_rotary: bool = False, name: str = "SCARA") -> SerialChain:
    """
    Build a SCARA manipulator.

    Default is the classic 3-DOF planar variant (R–R–P), with an optional 4th
    yaw wrist (R–R–P–R). The prismatic joint is aligned with +z.

    DH parameterization (standard):
        Link 1: a=0,   α=0,    d=0,     θ = q1
        Link 2: a=l1,  α=0,    d=0,     θ = q2
        Link 3: a=l2,  α=0,    d = d_home + q3 (prismatic)
        Link 4 (optional wrist yaw): a=0, α=0, d=0, θ = q4

    Parameters
    ----------
    l1, l2 : float
        Planar link lengths.
    d_home : float
        Home offset for the vertical prismatic joint (positive along +z).
    wrist_rotary : bool
        If True, include a 4th revolute wrist yaw (R–R–P–R).
    name : str
        Model name.

    Returns
    -------
    SerialChain
    """
    links: List[DHLink] = [
        DHLink(a=0.0, alpha=0.0, d=0.0, joint_type="R"),          # q1
        DHLink(a=l1, alpha=0.0, d=0.0, joint_type="R"),            # q2
        DHLink(a=l2, alpha=0.0, d=d_home, joint_type="P"),         # q3 (prismatic along +z)
    ]
    if wrist_rotary:
        links.append(DHLink(a=0.0, alpha=0.0, d=0.0, joint_type="R"))  # q4

    return SerialChain(links, M=np.eye(4), name=name if not wrist_rotary else f"{name}_4dof")


# --------------------------- Spherical wrists ----------------------------- #

def spherical_wrist(*, wrist_type: int, d7: float = 0.0, name: str = "wrist") -> SerialChain:
    """
    Build a 3-DOF spherical wrist preset with tool offset ``Tz(d7)`` at the tip.

    The three types differ by the **effective** rotation_kinematics order, realized via DH
    link twists (α) that reorient each joint's local z-axis:

        Type 1 (Roll–Pitch–Roll ≈ Z–X–Z): α = [-π/2, +π/2, 0]
        Type 2 (Roll–Pitch–Yaw ≈ Z–Y–Z):  α = [+π/2, -π/2, 0]
        Type 3 (Pitch–Yaw–Roll ≈ X–Y–Z):  α = [-π/2, -π/2, 0]

    All links have a=0 and d=0; the tool offset is applied as M = Tz(d7).

    Parameters
    ----------
    wrist_type : {1, 2, 3}
        Select rotation_kinematics order preset (see above).
    d7 : float
        Tool offset along final +z.
    name : str
        Model name.

    Returns
    -------
    SerialChain
    """
    if wrist_type not in (1, 2, 3):
        raise ValueError("wrist_type must be one of {1, 2, 3}")

    if wrist_type == 1:                 # Z–X–Z (roll–pitch–roll)
        alphas = (-np.pi / 2, +np.pi / 2, 0.0)
    elif wrist_type == 2:               # Z–Y–Z (roll–pitch–yaw variant)
        alphas = (+np.pi / 2, -np.pi / 2, 0.0)
    else:                               # wrist_type == 3 → X–Y–Z
        alphas = (-np.pi / 2, -np.pi / 2, 0.0)

    L1 = DHLink(a=0.0, alpha=alphas[0], d=0.0, joint_type="R")  # q4 (about z1 aligned as desired)
    L2 = DHLink(a=0.0, alpha=alphas[1], d=0.0, joint_type="R")  # q5
    L3 = DHLink(a=0.0, alpha=alphas[2], d=0.0, joint_type="R")  # q6

    M = _tool_Tz(d7)
    return SerialChain([L1, L2, L3], M=M, name=f"{name}_type{wrist_type}")


# --------------------------- Convenience Aliases -------------------------- #

# Backwards/alternate names if you prefer a slightly different API surface:
build_scara = scara
build_spherical_wrist = spherical_wrist
build_planar_2r = planar_2r
