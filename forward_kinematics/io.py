# forward_kinematics/io.py
"""
I/O, schema, and builders for forward_kinematics-kinematics robot specifications.

Features
--------
- JSON **and** YAML input (auto-detected by file extension).
- JSON Schema validation (single source of truth).
- Builders for DH, MDH, and PoE serial chains.
- Simple transform saver for 4×4 matrices.

Spec format (uniform across JSON/YAML)
--------------------------------------
Top-level keys:
  name   : str (optional)
  format : "dh" | "mdh" | "poe"        # alias: "method"
  M      : 4x4 matrix (optional)       # tool/home transform
  links  : list of link dicts          # see each format below

DH/MDH link:
  { "a": float, "alpha": float, "d": float,
    "theta_offset": float=0.0, "joint_type": "R"|"P" }

PoE link (space frame screw):
  { "omega": [wx, wy, wz], "v": [vx, vy, vz] }

Examples
--------
DH JSON:
{
  "name": "planar_2r",
  "format": "dh",
  "links": [
    {"a": 0.0, "alpha": 0.0, "d": 0.0, "joint_type": "R"},
    {"a": 1.0, "alpha": 0.0, "d": 0.0, "joint_type": "R"}
  ],
  "M": [[1,0,0,1.0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
}

PoE YAML:
name: wrist
format: poe
M: [[1,0,0,0],[0,1,0,0],[0,0,1,0.15],[0,0,0,1]]
links:
  - {omega: [0,0,1], v: [0,0,0]}
  - {omega: [1,0,0], v: [0,0,0]}
  - {omega: [0,0,1], v: [0,0,0]}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import json
import math
import numbers

import numpy as np

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:  # pragma: no cover
    yaml = None
    _HAVE_YAML = False

from .core import DHLink, MDHLink, PoELink, SerialChain


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def robot_schema() -> Dict[str, Any]:
    """Return the JSON Schema that validates our robot specification."""
    # Minimal but strict on key shapes; numbers only, arrays sized where critical.
    number = {"type": "number"}
    vec3 = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    mat4x4 = {
        "type": "array",
        "minItems": 4,
        "maxItems": 4,
        "items": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": number,
        },
    }
    dh_link = {
        "type": "object",
        "required": ["a", "alpha", "d"],
        "properties": {
            "a": number,
            "alpha": number,
            "d": number,
            "theta_offset": number,
            "joint_type": {"type": "string", "enum": ["R", "P", "r", "p"]},
        },
        "additionalProperties": False,
    }
    poe_link = {
        "type": "object",
        "required": ["omega", "v"],
        "properties": {
            "omega": vec3,
            "v": vec3,
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Forward Kinematics Robot Specification",
        "type": "object",
        "required": ["format", "links"],
        "properties": {
            "name": {"type": "string"},
            "format": {"type": "string", "enum": ["dh", "mdh", "poe"]},
            # Back-compat alias some folks use:
            "method": {"type": "string", "enum": ["dh", "mdh", "poe"]},
            "M": mat4x4,
            "links": {
                "type": "array",
                "minItems": 1,
                "items": {"oneOf": [dh_link, poe_link]},
            },
        },
        "additionalProperties": False,
        "allOf": [
            {
                "if": {
                    "properties": {"format": {"const": "poe"}}
                },
                "then": {
                    "properties": {"links": {"items": poe_link}}
                },
                "else": {
                    "properties": {"links": {"items": dh_link}}
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_spec(spec: Mapping[str, Any], schema: Optional[Mapping[str, Any]] = None) -> None:
    """
    Validate a spec mapping against the JSON Schema.

    Raises
    ------
    jsonschema.ValidationError
        If the spec does not conform.
    """
    try:
        import jsonschema  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("jsonschema is required for validation. Install with `pip install jsonschema`.") from exc

    schema = schema or robot_schema()
    jsonschema.validate(instance=dict(spec), schema=dict(schema))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _is_yaml_path(path: Union[str, Path]) -> bool:
    p = Path(path)
    return p.suffix.lower() in {".yaml", ".yml"}


def load_spec_from_file(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a robot spec from JSON or YAML file (auto-detected by extension).

    Returns
    -------
    dict
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    text = p.read_text(encoding="utf-8")
    if _is_yaml_path(p):
        if not _HAVE_YAML:
            raise ImportError("PyYAML is required to read YAML. Install with `pip install pyyaml`.")
        data = yaml.safe_load(text)  # type: ignore
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Top-level spec must be a mapping/dict.")
    return data


def load_spec(path_or_dict: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Load a spec from a file path (JSON/YAML) **or** return a validated copy of
    an in-memory dict (no validation applied here).
    """
    if isinstance(path_or_dict, (str, Path)):
        return load_spec_from_file(path_or_dict)
    # Shallow copy to detach from caller
    return dict(path_or_dict)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _coerce_4x4(M: Any) -> np.ndarray:
    A = np.asarray(M, dtype=float)
    if A.shape != (4, 4):
        raise ValueError(f"M must be 4x4, got shape {A.shape}")
    return A


def _format_from_spec(spec: Mapping[str, Any]) -> str:
    fmt = spec.get("format") or spec.get("method")
    if not isinstance(fmt, str):
        raise ValueError("Spec must include 'format' (or alias 'method').")
    f = fmt.strip().lower()
    if f not in {"dh", "mdh", "poe"}:
        raise ValueError(f"Unsupported format: {fmt}")
    return f


def build_chain_from_spec(spec: Mapping[str, Any]) -> SerialChain:
    """
    Build a :class:`SerialChain` from a validated spec mapping.

    Notes
    -----
    - This function does not perform schema validation; call
      :func:`validate_spec` first if you need explicit checking.
    """
    fmt = _format_from_spec(spec)
    links_data = spec.get("links")
    if not isinstance(links_data, list) or not links_data:
        raise ValueError("'links' must be a non-empty list.")

    name = str(spec.get("name") or f"{fmt}_robot")
    M = spec.get("M", None)
    M_mat = _coerce_4x4(M) if M is not None else np.eye(4)

    if fmt == "poe":
        links: List[PoELink] = []
        for i, ld in enumerate(links_data):
            try:
                w = ld["omega"]
                v = ld["v"]
            except KeyError as exc:
                raise KeyError(f"PoE link #{i} missing key {exc}.") from exc
            links.append(PoELink(np.array(w, float), np.array(v, float)))
        return SerialChain(links, M=M_mat, name=name)

    # DH/MDH path
    dh_cls = DHLink if fmt == "dh" else MDHLink
    links_dh: List[DHLink] = []
    for i, ld in enumerate(links_data):
        try:
            a = float(ld["a"])
            alpha = float(ld["alpha"])
            d = float(ld["d"])
            theta_offset = float(ld.get("theta_offset", 0.0))
            jt = str(ld.get("joint_type", "R")).upper()
            if jt not in {"R", "P"}:
                raise ValueError("joint_type must be 'R' or 'P'")
        except KeyError as exc:
            raise KeyError(f"DH/MDH link #{i} missing key {exc}.") from exc
        links_dh.append(dh_cls(a=a, alpha=alpha, d=d, theta_offset=theta_offset, joint_type=jt))

    return SerialChain(links_dh, M=M_mat, name=name)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_transform_json(path: Union[str, Path], T: np.ndarray) -> None:
    """
    Save a 4×4 homogeneous transform as JSON { "matrix": [[...],[...],[...],[...]] }.
    """
    A = np.asarray(T, dtype=float)
    if A.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got shape {A.shape}")
    Path(path).write_text(json.dumps({"matrix": A.tolist()}, indent=2), encoding="utf-8")


__all__ = [
    "robot_schema",
    "validate_spec",
    "load_spec_from_file",
    "load_spec",
    "build_chain_from_spec",
    "save_transform_json",
]
