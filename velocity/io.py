# velocity/io.py
"""
File I/O helpers for the Velocity Kinematics Toolkit.

What’s provided
---------------
- load_dh_from_file(path)   -> dict with {"name", "joints", "tool"?}
- load_urdf_from_file(path) -> dict with {"name", "raw"?, "links"?, "joints"?}
- save_report(path, obj)    -> JSON writer tolerant of NumPy types

Design goals
------------
- Minimal, dependency-light. YAML/URDF parsing is optional.
- Deterministic, test-friendly pure functions (no globals, no side effects).
- Clear validation of DH specs with readable error messages.

DH schema (standard Craig DH)
-----------------------------
{
  "name": "my_arm",               # optional
  "joints": [
    {"name":"j1", "type":"R", "alpha":0.0, "a":0.5, "d":0.0, "theta":0.0},
    ...
  ],
  "tool": { "xyz": [0.0, 0.0, 0.0] }   # optional; or a full 4x4 homogeneous matrix
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np


# ------------------------------- Public API ---------------------------------- #

def load_dh_from_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a standard-DH robot description from YAML or JSON.

    Parameters
    ----------
    path : str | Path
        .yml/.yaml or .json file

    Returns
    -------
    dict
        Validated DH dictionary with at least a "joints" list.

    Raises
    ------
    ValueError
        If schema is invalid or file unreadable.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"DH file not found: {p}")

    data = _read_yaml_or_json(p)
    _validate_dh_dict(data, source=str(p))
    return data


def load_urdf_from_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a URDF model. If `urdfpy` is available, parse structure into a dict; otherwise,
    return a minimal dict with raw text so higher layers can decide what to do.

    Parameters
    ----------
    path : str | Path
        .urdf or .xml file

    Returns
    -------
    dict
        At minimum: {"name": "<stem>", "raw": "<string>"} if no parser is installed.
        If `urdfpy` is present: {"name", "links", "joints"} with basic fields.

    Notes
    -----
    The core `URDFRobot` currently raises a runtime error until you integrate
    a backend (pinocchio/urdfpy). This function exists so your CLI can accept
    URDF inputs gracefully and fail with a clear message in the kinematics step.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"URDF file not found: {p}")
    text = p.read_text(encoding="utf-8", errors="ignore")

    try:
        from urdfpy import URDF  # type: ignore
        robot = URDF.load(str(p))
        links = [{"name": lk.name} for lk in robot.links]
        joints = [{"name": j.name, "type": j.joint_type, "parent": j.parent, "child": j.child} for j in robot.joints]
        name = robot.name or p.stem
        return {"name": name, "links": links, "joints": joints}
    except Exception:
        # Soft fallback: return raw so callers can present a friendly message.
        return {"name": p.stem, "raw": text}


def save_report(path: str | Path, obj: Any) -> None:
    """
    Save a JSON report to `path`, handling NumPy arrays and numbers.

    Parameters
    ----------
    path : str | Path
        Output file path (directories are created if needed).
    obj : Any
        Serializable object; NumPy arrays/numbers are converted automatically.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json_dumps(obj), encoding="utf-8")


# ------------------------------ Internal utils -------------------------------- #

def _read_yaml_or_json(path: Path) -> Dict[str, Any]:
    sfx = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if sfx in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ValueError(
                f"Reading YAML requires PyYAML. Install with `pip install pyyaml`. File: {path}"
            ) from e
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping: {path}")
        return data
    elif sfx == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"JSON root must be an object: {path}")
        return data
    else:
        raise ValueError(f"Unsupported file extension for {path.name} (use .yml/.yaml or .json)")


def _validate_dh_dict(data: Mapping[str, Any], source: str = "<dict>") -> None:
    # Top-level
    if "joints" not in data or not isinstance(data["joints"], list) or len(data["joints"]) == 0:
        raise ValueError(f"{source}: missing or empty 'joints' list.")

    # Joint entries
    for i, jd in enumerate(data["joints"]):
        if not isinstance(jd, Mapping):
            raise ValueError(f"{source}: joint[{i}] must be a mapping.")
        for key in ("type", "alpha", "a"):
            if key not in jd:
                raise ValueError(f"{source}: joint[{i}] missing required key '{key}'.")
        jtype = str(jd["type"]).upper()
        if jtype not in {"R", "P"}:
            raise ValueError(f"{source}: joint[{i}].type must be 'R' or 'P'.")
        # Numeric coercion/validation
        for numkey in ("alpha", "a", "d", "theta"):
            if numkey in jd:
                try:
                    float(jd[numkey])  # raises ValueError if not numeric
                except Exception:
                    raise ValueError(f"{source}: joint[{i}].{numkey} must be numeric.")

    # Tool (optional): either {"xyz":[x,y,z]} or 4x4 homogeneous
    if "tool" in data:
        tool = data["tool"]
        if isinstance(tool, Mapping) and "xyz" in tool:
            xyz = tool["xyz"]
            if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
                raise ValueError(f"{source}: tool.xyz must be a length-3 list/tuple.")
            _ = [float(v) for v in xyz]  # numeric check
        elif isinstance(tool, (list, tuple)):
            arr = np.asarray(tool, dtype=float)
            if arr.shape != (4, 4):
                raise ValueError(f"{source}: tool 4x4 matrix expected, got {arr.shape}.")
        elif isinstance(tool, np.ndarray):
            if tool.shape != (4, 4):
                raise ValueError(f"{source}: tool 4x4 numpy array expected, got {tool.shape}.")
        else:
            raise ValueError(f"{source}: unsupported 'tool' format.")


def _json_dumps(obj: Any) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return str(o)
    return json.dumps(obj, indent=2, default=default)
