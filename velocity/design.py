# velocity/design.py
"""
Design presets + diagram helpers for the Velocity Kinematics Toolkit.

This module now serves TWO roles:
1) **Kinematic presets (DH-based)** for velocity analyses:
   - planar_2r, arm_3r_articulated, spherical_wrist(type=1..3), six_dof_spherical
   - utilities: jacobian_columns(), is_spherical_wrist_zero_block()
2) **Diagram helpers** (Mermaid, pyreverse, autoscan) retained from the
   initial version so existing tests keep working.

Standard DH (Craig):  Tz(d) · Rz(θ) · Tx(a) · Rx(α)
"""

from __future__ import annotations

# --------------------------- stdlib & typing ----------------------------------
import inspect
import io
from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

# ------------------------------ third party -----------------------------------
import numpy as np

# ------------------------------- local imports --------------------------------
from .core import JointDH, DHRobot
from .utils import transl

# ==============================================================================
# PART A — KINEMATIC PRESETS (velocity-focused)
# ==============================================================================

# ---- small helpers for tool transforms ---------------------------------------

def _tool_Tz(d: float) -> np.ndarray:
    """SE(3) pure translation along z."""
    T = np.eye(4)
    T[2, 3] = float(d)
    return T

def _tool_Tx(x: float) -> np.ndarray:
    """SE(3) pure translation along x."""
    T = np.eye(4)
    T[0, 3] = float(x)
    return T


# ---- Planar 2R ---------------------------------------------------------------

def planar_2r(l1: float, l2: float, *, name: str = "planar_2R") -> DHRobot:
    """
    Build a 2R planar arm in x–y (z out of plane). Correct standard-DH parameterization:

      L1: a=l1, α=0, d=0, type='R'
      L2: a=l2, α=0, d=0, type='R'
      tool: I

    FK:  T = Rz(q1) · Tx(l1) · Rz(q2) · Tx(l2)

    Returns
    -------
    DHRobot
    """
    j1 = JointDH(name="j1", joint_type="R", alpha=0.0, a=float(l1), d=0.0, theta=0.0)
    j2 = JointDH(name="j2", joint_type="R", alpha=0.0, a=float(l2), d=0.0, theta=0.0)
    return DHRobot([j1, j2], tool=np.eye(4), name=name)


# ---- 3R articulated arm (position subproblem) --------------------------------

def arm_3r_articulated(
    l1: float,
    l2: float,
    d3: float = 0.0,
    *,
    name: str = "arm_3R",
) -> DHRobot:
    """
    Shoulder–elbow–wrist-center 3R chain, convenient for decoupled analyses.

    Geometry:
      L1: a=l1, α=0, d=0, type='R'
      L2: a=l2, α=0, d=0, type='R'
      L3: a=0,  α=0, d=d3, type='R'  (optional constant offset on the third joint)
      tool: I
    """
    j1 = JointDH("j1", "R", 0.0, float(l1), 0.0, 0.0)
    j2 = JointDH("j2", "R", 0.0, float(l2), 0.0, 0.0)
    j3 = JointDH("j3", "R", 0.0, 0.0, float(d3), 0.0)
    return DHRobot([j1, j2, j3], tool=np.eye(4), name=name)


# ---- 3R spherical wrists (type presets) --------------------------------------

def spherical_wrist(*, wrist_type: int, d_tool: float = 0.0, name: str = "wrist") -> DHRobot:
    """
    Build a 3R spherical wrist preset with a tool offset Tz(d_tool).

    The three types differ by α that reorient local z-axes (effective rotation order):
      Type 1 (≈ Z–X–Z): α = [-π/2, +π/2, 0]
      Type 2 (≈ Z–Y–Z): α = [+π/2, -π/2, 0]
      Type 3 (≈ X–Y–Z): α = [-π/2, -π/2, 0]

    All links: a=0, d=0, type='R'
    """
    if wrist_type not in (1, 2, 3):
        raise ValueError("wrist_type must be one of {1, 2, 3}")
    if wrist_type == 1:
        alphas = (-np.pi/2, +np.pi/2, 0.0)
    elif wrist_type == 2:
        alphas = (+np.pi/2, -np.pi/2, 0.0)
    else:
        alphas = (-np.pi/2, -np.pi/2, 0.0)

    j4 = JointDH("j4", "R", alphas[0], 0.0, 0.0, 0.0)
    j5 = JointDH("j5", "R", alphas[1], 0.0, 0.0, 0.0)
    j6 = JointDH("j6", "R", alphas[2], 0.0, 0.0, 0.0)
    tool = _tool_Tz(float(d_tool))
    return DHRobot([j4, j5, j6], tool=tool, name=f"{name}_type{wrist_type}")


# ---- 6R = 3R arm + 3R wrist (classic wrist-decoupled structure) --------------

def six_dof_spherical(
    l1: float,
    l2: float,
    *,
    wrist_type: int = 1,
    d_tool: float = 0.0,
    name: str = "six_dof_spherical",
) -> DHRobot:
    """
    Compose a 6R manipulator as (3R arm) + (3R spherical wrist). The last three
    axes intersect by construction, enabling the classic "zero upper-right block"
    pattern in J for appropriate frames.
    """
    arm = arm_3r_articulated(l1, l2, name=f"{name}_arm3R")
    wrist = spherical_wrist(wrist_type=wrist_type, d_tool=d_tool, name=f"{name}_wrist3R")
    # Concatenate joints; tool combines (arm.tool is I)
    joints = arm.joints + wrist.joints
    tool = arm.tool @ wrist.tool
    return DHRobot(joints, tool=tool, name=name)


# ---- Velocity-kinematics utilities -------------------------------------------

def jacobian_columns(robot: DHRobot, q: Sequence[float]) -> List[np.ndarray]:
    """
    Return the list of Jacobian-generating columns c_i(q) for the given robot/q.

    For revolute i:  c_i = [ k_i × (p_e - p_i) ; k_i ]
    For prismatic i: c_i = [ k_i ; 0 ]
    """
    J = robot.jacobian_geometric(np.asarray(q, dtype=float))
    return [J[:, i].copy() for i in range(J.shape[1])]


def is_spherical_wrist_zero_block(robot: DHRobot, q: Sequence[float], atol: float = 1e-10) -> bool:
    """
    Check the "zero upper-right 3×3 block" pattern for a spherical wrist:
      Partition J = [[A, 0], [C, D]] with the last three joints forming the wrist.
      This function verifies J[:3, -3:] ≈ 0.
    """
    J = robot.jacobian_geometric(np.asarray(q, dtype=float))
    if J.shape[1] < 6:
        return False
    return np.allclose(J[:3, -3:], 0.0, atol=atol)


# ==============================================================================
# PART B — DIAGRAM HELPERS (kept from the original design.py)
# ==============================================================================

# -------- Mermaid diagram generation ------------------------------------------

def _typename(tp: Any) -> str:
    try:
        return tp.__name__  # type: ignore[attr-defined]
    except Exception:
        s = str(tp)
        return s.replace("typing.", "")

def _class_attrs(cls: type) -> List[Tuple[str, str]]:
    attrs: List[Tuple[str, str]] = []
    if is_dataclass(cls):
        for f in fields(cls):
            attrs.append((f.name, _typename(f.type)))
        return attrs
    ann = getattr(cls, "__annotations__", {}) or {}
    for name, tp in ann.items():
        if name.startswith("_"):
            continue
        attrs.append((name, _typename(tp)))
    for name, val in vars(cls).items():
        if name.startswith("_"):
            continue
        if callable(val):
            continue
        if name not in {n for n, _ in attrs}:
            attrs.append((name, _typename(type(val))))
    return attrs

def _mermaid_block_for_class(cls: type) -> str:
    lines = [f"class {cls.__name__} {{"]

    # attributes
    attrs = _class_attrs(cls)
    for name, tname in attrs:
        lines.append(f"  +{name} : {tname}")

    # public methods
    methods = []
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        try:
            sig = str(inspect.signature(member))
        except Exception:
            sig = "()"
        methods.append(f"  +{name}{sig}")
    if methods:
        if attrs:
            lines.append("  --")
        lines.extend(methods)
    lines.append("}")
    return "\n".join(lines)

def mermaid_from_classes(
    classes: Sequence[type],
    relations: Optional[Sequence[Tuple[str, str, str]]] = None,
    outfile: Optional[str | Path] = None,
    title: str = "Velocity Toolkit – Class Diagram",
) -> str:
    out = io.StringIO()
    print(f"%% {title}", file=out)
    print("```mermaid", file=out)
    print("classDiagram", file=out)
    for cls in classes:
        print(_mermaid_block_for_class(cls), file=out)
    if relations:
        for src, dst, label in relations:
            label_txt = f" : {label}" if label else ""
            print(f"{src} --> {dst}{label_txt}", file=out)
    print("```", file=out)
    text = out.getvalue()
    if outfile:
        p = Path(outfile)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return text


# -------- autoscan (shallow) --------------------------------------------------

def autoscan_package(
    package_module,
    include_private: bool = False,
    max_depth: int = 1,
) -> List[type]:
    classes: List[type] = []
    def _walk(mod, depth: int) -> None:
        if depth < 0:
            return
        for name, member in vars(mod).items():
            if not include_private and name.startswith("_"):
                continue
            if inspect.isclass(member) and member.__module__.startswith(mod.__name__):
                classes.append(member)
            if depth > 0 and inspect.ismodule(member) and member.__name__.startswith(mod.__name__):
                try:
                    _walk(member, depth - 1)
                except Exception:
                    continue
    _walk(package_module, max_depth)
    seen = set()
    uniq = []
    for c in classes:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


# -------- pyreverse bridge (optional dependency) ------------------------------

def run_pyreverse(package_dir: str | Path, outdir: str | Path) -> dict:
    try:
        from pylint.pyreverse.main import Run  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pyreverse (from pylint) is required. Install with `pip install pylint`."
        ) from e

    pkg = str(Path(package_dir))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Run([pkg, "-o", "dot", " -A", "-S", "-p", "velocity"], exit=False)
    Run([pkg, "-o", "plantuml", "-p", "velocity"], exit=False)

    moved = {}
    for name in ("classes.dot", "packages.dot", "classes.uml", "packages.uml"):
        src = Path.cwd() / name
        if src.exists():
            dest = outdir / name
            dest.write_bytes(src.read_bytes())
            try:
                src.unlink()
            except Exception:
                pass
            moved[name.replace(".", "_")] = str(dest)
    if "classes_dot" not in moved and "classes_uml" not in moved:
        raise RuntimeError("pyreverse did not produce expected outputs.")
    return moved


# -------- convenience one-shot Mermaid ----------------------------------------

def default_mermaid(outfile: str | Path) -> str:
    """
    Emit a curated Mermaid diagram of velocity core classes.
    """
    from . import core  # local import to avoid circulars during init
    return mermaid_from_classes(
        classes=[core.JointDH, core.DHRobot, core.solvers],
        relations=[("DHRobot", "JointDH", "contains *")],
        outfile=outfile,
        title="Velocity – Core Classes",
    )
