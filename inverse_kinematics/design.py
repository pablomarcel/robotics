# inverse_kinematics/design.py
"""
Preset builders for **inverse_kinematics-kinematics** workflows.

These presets construct kinematic chains (using **standard DH**) that are
convenient for testing analytic and iterative IK across Chapter 6:

- 2R planar arm (closed-form IK per 6.10–6.22)
- 3R articulated arm (position subproblem used in wrist decoupling)
- 3-DOF spherical wrists (types 1–3 by α-patterns)
- 6R arm with spherical wrist (classic Pieper-solvable structure)

All presets return :class:`inverse_kinematics.core.SerialChain` instances so you can feed
them directly to the solvers in :mod:`inverse_kinematics.core` or through the façade in
:mod:`inverse_kinematics.app`.

Design notes
------------
* Standard DH (Craig):  Tz(d) · Rz(θ) · Tx(a) · Rx(α).
* For planar links (α = d = 0), each link contributes Rz(qi) · Tx(ai).
* Tool offsets are only used where they represent actual physical TCP offsets,
  not to “store” link lengths (which must live in `a` of the corresponding link).
"""

from __future__ import annotations

from typing import List

import numpy as np

from .core import DHLink, SerialChain, Transform  # noqa: F401 (Transform referenced in docstrings)


# ------------------------------ Utilities --------------------------------- #

def _tool_Tz(d: float) -> np.ndarray:
    """Return a 4×4 homogeneous transform representing a pure translation Tz(d)."""
    M = np.eye(4)
    M[2, 3] = float(d)
    return M


def _tool_Tx(x: float) -> np.ndarray:
    """Return a 4×4 homogeneous transform representing a pure translation Tx(x)."""
    M = np.eye(4)
    M[0, 3] = float(x)
    return M


# ------------------------------- Planar 2R -------------------------------- #

def planar_2r(l1: float, l2: float, *, name: str = "planar_2R") -> SerialChain:
    """
    Build a **planar 2R** arm in the x–y plane (z is out of plane).

    Correct standard-DH parameterization for a classic 2R:
        - L1: a1 = l1, α1 = 0, d1 = 0, joint_type="R"
        - L2: a2 = l2, α2 = 0, d2 = 0, joint_type="R"
        - M  : identity (no fake length in the tool transform)

    With α = d = 0 for both links, FK is:
        T = Rz(q1) · Tx(l1) · Rz(q2) · Tx(l2)

    Parameters
    ----------
    l1, l2 : float
        Link lengths.

    Returns
    -------
    SerialChain
    """
    L1 = DHLink(a=float(l1), alpha=0.0, d=0.0, joint_type="R")  # q1
    L2 = DHLink(a=float(l2), alpha=0.0, d=0.0, joint_type="R")  # q2
    return SerialChain([L1, L2], M=np.eye(4), name=name)


# ---------------------------- 3R articulated arm -------------------------- #

def arm_3r_articulated(
    l1: float,
    l2: float,
    d3: float = 0.0,
    *,
    name: str = "arm_3R",
) -> SerialChain:
    """
    Build a **3R articulated arm** (shoulder–elbow–wrist center) suitable as the
    *position* subproblem in spherical-wrist decoupling.

    Geometry (planar shoulder & elbow, third revolute about the wrist-center axis):
        - L1: base yaw,     a1 = l1, α1 = 0, d1 = 0
        - L2: elbow pitch,  a2 = l2, α2 = 0, d2 = 0
        - L3: wrist-center, a3 = 0,  α3 = 0, d3 = d3 (optional constant offset)
        - M : identity (wrist center coincides with tip of link 2 when d3 = 0)

    This keeps the wrist center at the end of L2 for d3=0, while still allowing
    a fixed z-offset if needed via `d3`.

    Parameters
    ----------
    l1 : float
        First planar link length (shoulder→elbow).
    l2 : float
        Second planar link length (elbow→wrist center).
    d3 : float, optional
        Optional constant offset along +z for the third joint.

    Returns
    -------
    SerialChain
    """
    L1 = DHLink(a=float(l1), alpha=0.0, d=0.0, joint_type="R")  # q1 (base yaw)
    L2 = DHLink(a=float(l2), alpha=0.0, d=0.0, joint_type="R")  # q2 (elbow)
    L3 = DHLink(a=0.0,       alpha=0.0, d=float(d3), joint_type="R")  # q3 (wrist-center rot)
    return SerialChain([L1, L2, L3], M=np.eye(4), name=name)


# --------------------------- Spherical wrists (3R) ------------------------ #

def spherical_wrist(*, wrist_type: int, d_tool: float = 0.0, name: str = "wrist") -> SerialChain:
    """
    Build a **3-DOF spherical wrist** preset with a tool offset ``Tz(d_tool)``.

    The three types differ by the **effective** rotation_kinematics order, realized via DH
    link twists (α) that reorient each joint's local z-axis:

        Type 1 (Roll–Pitch–Roll ≈ Z–X–Z): α = [-π/2, +π/2, 0]
        Type 2 (Roll–Pitch–Yaw  ≈ Z–Y–Z): α = [+π/2, -π/2, 0]
        Type 3 (Pitch–Yaw–Roll  ≈ X–Y–Z): α = [-π/2, -π/2, 0]

    All links have a=0 and d=0; the tool offset is applied as M = Tz(d_tool).

    Parameters
    ----------
    wrist_type : {1, 2, 3}
        Select rotation_kinematics-order preset (see above).
    d_tool : float
        Tool offset along final +z (distance from wrist center to TCP).
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

    L4 = DHLink(a=0.0, alpha=alphas[0], d=0.0, joint_type="R")  # q4
    L5 = DHLink(a=0.0, alpha=alphas[1], d=0.0, joint_type="R")  # q5
    L6 = DHLink(a=0.0, alpha=alphas[2], d=0.0, joint_type="R")  # q6

    M = _tool_Tz(float(d_tool))
    return SerialChain([L4, L5, L6], M=M, name=f"{name}_type{wrist_type}")


# ------------------------ 6R with spherical wrist (Pieper) ---------------- #

def six_dof_spherical(
    l1: float,
    l2: float,
    *,
    wrist_type: int = 1,
    d_tool: float = 0.0,
    name: str = "six_dof_spherical",
) -> SerialChain:
    """
    Compose a **6R** manipulator as (3R arm) + (3R spherical wrist).

    This is the classic Pieper-solvable architecture used for analytic IK with
    wrist decoupling (Chapter 6: (6.4–6.9), (6.125), (6.137–6.140)).

    Parameters
    ----------
    l1, l2 : float
        Arm link lengths (shoulder→elbow, elbow→wrist center).
    wrist_type : {1, 2, 3}
        Spherical wrist rotation_kinematics-order preset (see :func:`spherical_wrist`).
    d_tool : float
        Tool offset along wrist +z (TCP distance from wrist center).
    name : str
        Model name.

    Returns
    -------
    SerialChain
        A 6-DOF chain where the last three joint axes intersect and are
        orthogonal (by construction from the wrist preset).
    """
    arm = arm_3r_articulated(l1, l2, name=f"{name}_arm3R")  # M = I
    wrist = spherical_wrist(wrist_type=wrist_type, d_tool=d_tool, name=f"{name}_wrist3R")

    # Concatenate links; final tool is arm.M @ wrist.M = wrist.M (since arm.M = I)
    links: List[DHLink] = []
    links.extend(arm.links())
    links.extend(wrist.links())

    M = arm.M @ wrist.M
    return SerialChain(links, M=M, name=name)


# --------------------------- Convenience Aliases -------------------------- #

build_planar_2r = planar_2r
build_arm_3r = arm_3r_articulated
build_spherical_wrist = spherical_wrist
build_six_dof_spherical = six_dof_spherical
