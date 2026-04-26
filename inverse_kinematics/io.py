# inverse_kinematics/io.py
"""
I/O, schema, and builders for **inverse_kinematics-kinematics problems**.

This module complements the forward_kinematics package's I/O by focusing on the
*inverse_kinematics* side. It supports two levels of modeling input:

1) **High-level models** (recommended for IK):
   {
     "kind": "planar2r" | "arm_3r" | "spherical_wrist" | "six_dof_spherical",
     ... parameters per kind ...
   }

2) **Low-level chains** (DH/MDH), identical to forward_kinematics core:
   {
     "format": "dh" | "mdh",
     "links": [ {a, alpha, d, theta_offset?, joint_type?}, ... ],
     "M": [[...],[...],[...],[...]]   # optional tool transform
   }

A complete IK *problem* file is a JSON/YAML object with:
  {
    "model":  <model spec, high-level or low-level>,
    "method": {"method": "analytic" | "iterative", ...params...},
    "pose":   {"x":..., "y":..., "z"?, "roll"?, "pitch"?, "yaw"?}  OR
              {"T": [[...],[...],[...],[...]]}
  }

Features
--------
- JSON **and** YAML input (auto-detected by extension).
- JSON Schema validation for IK problems and low-level chains.
- Builders for DH/MDH serial chains and high-level presets (via design.py).
- Helpers to save solution sets to JSON.

Examples
--------
IK problem (high-level model):
{
  "model":  {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
  "method": {"method": "analytic"},
  "pose":   {"x": 1.0, "y": 1.0}
}

IK problem (low-level chain):
{
  "model": {
    "format": "dh",
    "links": [
      {"a": 0.0, "alpha": 0.0, "d": 0.0, "joint_type": "R"},
      {"a": 1.0, "alpha": 0.0, "d": 0.0, "joint_type": "R"}
    ],
    "M": [[1,0,0,1.0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  },
  "method": {"method": "iterative", "tol": 1e-8, "itmax": 200, "lambda": 1e-3},
  "pose":   {"T": [[...],[...],[...],[...]]}
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import json
import numpy as np

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:  # pragma: no cover
    yaml = None
    _HAVE_YAML = False

from .core import DHLink, MDHLink, SerialChain
from . import design as design_mod


PathLike = Union[str, Path]


# -----------------------------------------------------------------------------
# JSON Schemas
# -----------------------------------------------------------------------------

def _number() -> Dict[str, Any]:
    return {"type": "number"}


def _vec3() -> Dict[str, Any]:
    return {"type": "array", "items": _number(), "minItems": 3, "maxItems": 3}


def _mat4x4() -> Dict[str, Any]:
    return {
        "type": "array",
        "minItems": 4,
        "maxItems": 4,
        "items": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": _number(),
        },
    }


def _dh_link_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["a", "alpha", "d"],
        "properties": {
            "a": _number(),
            "alpha": _number(),
            "d": _number(),
            "theta_offset": _number(),
            "joint_type": {"type": "string", "enum": ["R", "P", "r", "p"]},
        },
        "additionalProperties": False,
    }


def chain_schema() -> Dict[str, Any]:
    """Schema for **low-level** DH/MDH chain models."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Inverse IK — Low-level Chain (DH/MDH)",
        "type": "object",
        "required": ["format", "links"],
        "properties": {
            "name": {"type": "string"},
            "format": {"type": "string", "enum": ["dh", "mdh"]},
            "links": {
                "type": "array",
                "minItems": 1,
                "items": _dh_link_schema(),
            },
            "M": _mat4x4(),
        },
        "additionalProperties": False,
    }


def model_schema() -> Dict[str, Any]:
    """Schema for **high-level** model specs used by IK presets."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Inverse IK — High-level Model",
        "type": "object",
        "required": ["kind"],
        "properties": {
            "kind": {"type": "string", "enum": [
                "planar2r", "arm_3r", "spherical_wrist", "six_dof_spherical"
            ]},
            # planar2r
            "l1": _number(),
            "l2": _number(),
            # arm_3r
            "d3": _number(),
            # spherical_wrist
            "wrist_type": {"type": "integer", "minimum": 1, "maximum": 3},
            "d_tool": _number(),
            # six_dof_spherical
            "d7": _number(),  # alias for d_tool if you prefer d7 naming
            "name": {"type": "string"},
        },
        "additionalProperties": True,  # allow future fields
    }


def method_schema() -> Dict[str, Any]:
    """Schema for IK method selection & parameters."""
    return {
        "type": "object",
        "required": ["method"],
        "properties": {
            "method": {"type": "string", "enum": ["analytic", "iterative"]},
            "tol": _number(),
            "itmax": {"type": "integer", "minimum": 1},
            "lambda": _number(),
        },
        "additionalProperties": True,
    }


def pose_schema() -> Dict[str, Any]:
    """Schema for pose targets: either (x,y[,z,roll,pitch,yaw]) or 4×4 T."""
    return {
        "oneOf": [
            {
                "type": "object",
                "required": ["x", "y"],
                "properties": {
                    "x": _number(),
                    "y": _number(),
                    "z": _number(),
                    "roll": _number(),
                    "pitch": _number(),
                    "yaw": _number(),
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": ["T"],
                "properties": {"T": _mat4x4()},
                "additionalProperties": False,
            },
        ]
    }


def problem_schema() -> Dict[str, Any]:
    """Top-level IK problem schema."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Inverse IK Problem",
        "type": "object",
        "required": ["model", "method", "pose"],
        "properties": {
            "model": {"oneOf": [model_schema(), chain_schema()]},
            "method": method_schema(),
            "pose": pose_schema(),
        },
        "additionalProperties": False,
    }


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def _validate(instance: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "jsonschema is required for validation. Install with `pip install jsonschema`."
        ) from exc
    jsonschema.validate(instance=dict(instance), schema=dict(schema))


def validate_problem(problem: Mapping[str, Any]) -> None:
    """Validate an IK problem dict against :func:`problem_schema`."""
    _validate(problem, problem_schema())


def validate_chain_model(model: Mapping[str, Any]) -> None:
    """Validate a low-level DH/MDH chain model against :func:`chain_schema`."""
    _validate(model, chain_schema())


def validate_high_level_model(model: Mapping[str, Any]) -> None:
    """Validate a high-level model against :func:`model_schema`."""
    _validate(model, model_schema())


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def _is_yaml_path(path: PathLike) -> bool:
    return Path(path).suffix.lower() in {".yaml", ".yml"}


def load_json_or_yaml(path: PathLike) -> Dict[str, Any]:
    """Load dict from a JSON or YAML file based on extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8")
    if _is_yaml_path(p):
        if not _HAVE_YAML:
            raise ImportError("PyYAML is required for YAML. Install with `pip install pyyaml`.")
        data = yaml.safe_load(text)  # type: ignore
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Top-level document must be a mapping/dict.")
    return data


def load_problem_from_file(path: PathLike, *, validate: bool = True) -> Dict[str, Any]:
    """Load an IK problem from JSON/YAML file."""
    data = load_json_or_yaml(path)
    if validate:
        validate_problem(data)
    return data


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------

def _coerce_4x4(M: Any) -> np.ndarray:
    A = np.asarray(M, dtype=float)
    if A.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {A.shape}")
    return A


def _format_from_model(model: Mapping[str, Any]) -> Optional[str]:
    fmt = model.get("format") or model.get("method")
    return str(fmt).strip().lower() if isinstance(fmt, str) else None


def build_chain_from_low_level(model: Mapping[str, Any]) -> SerialChain:
    """
    Build a :class:`SerialChain` from a **low-level** DH/MDH model mapping.
    """
    fmt = _format_from_model(model)
    if fmt not in {"dh", "mdh"}:
        raise ValueError("Low-level model must include 'format' of 'dh' or 'mdh'.")
    links_data = model.get("links")
    if not isinstance(links_data, list) or not links_data:
        raise ValueError("'links' must be a non-empty list.")

    name = str(model.get("name") or f"{fmt}_robot")
    M = model.get("M", None)
    M_mat = _coerce_4x4(M) if M is not None else np.eye(4)

    dh_cls = DHLink if fmt == "dh" else MDHLink
    links: List[DHLink] = []
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
        links.append(dh_cls(a=a, alpha=alpha, d=d, theta_offset=theta_offset, joint_type=jt))

    return SerialChain(links, M=M_mat, name=name)


def build_chain_from_high_level(model: Mapping[str, Any]) -> SerialChain:
    """
    Build a :class:`SerialChain` from a **high-level** model mapping
    (planar2r, arm_3r, spherical_wrist, six_dof_spherical).
    """
    kind = str(model.get("kind", "")).strip().lower()
    if not kind:
        raise ValueError("High-level model must include 'kind'.")

    if kind == "planar2r":
        l1 = float(model["l1"])
        l2 = float(model["l2"])
        return design_mod.planar_2r(l1, l2, name=str(model.get("name", "planar_2R")))
    if kind == "arm_3r":
        l1 = float(model["l1"])
        l2 = float(model["l2"])
        d3 = float(model.get("d3", 0.0))
        return design_mod.arm_3r_articulated(l1, l2, d3=d3, name=str(model.get("name", "arm_3R")))
    if kind == "spherical_wrist":
        wrist_type = int(model.get("wrist_type", 1))
        d_tool = float(model.get("d_tool", model.get("d7", 0.0)))
        return design_mod.spherical_wrist(wrist_type=wrist_type, d_tool=d_tool, name=str(model.get("name", "wrist")))
    if kind == "six_dof_spherical":
        l1 = float(model["l1"])
        l2 = float(model["l2"])
        wrist_type = int(model.get("wrist_type", 1))
        d_tool = float(model.get("d_tool", model.get("d7", 0.0)))
        return design_mod.six_dof_spherical(l1, l2, wrist_type=wrist_type, d_tool=d_tool,
                                            name=str(model.get("name", "six_dof_spherical")))

    raise ValueError(f"Unsupported high-level model kind: {kind}")


def build_chain_from_model(model: Mapping[str, Any], *, validate: bool = False) -> SerialChain:
    """
    Build a :class:`SerialChain` from either a high-level or low-level model dict.

    If `validate` is True, the function validates against the appropriate schema.
    """
    if "kind" in model:
        if validate:
            validate_high_level_model(model)
        return build_chain_from_high_level(model)
    if "format" in model or "method" in model:
        if validate:
            validate_chain_model(model)
        return build_chain_from_low_level(model)
    raise ValueError("Model dict must include either 'kind' (high-level) or 'format'/'method' (low-level).")


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

def save_solutions_json(path: PathLike, solutions: Sequence[np.ndarray]) -> None:
    """
    Save a set of IK solutions as JSON:
      { "solutions": [[...], [...], ...] }
    """
    arrs = [np.asarray(s, dtype=float).reshape(-1).tolist() for s in solutions]
    payload = {"solutions": arrs}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_problem_json(path: PathLike, problem: Mapping[str, Any]) -> None:
    """Save an IK problem dict to JSON (no validation performed here)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(problem), indent=2), encoding="utf-8")


__all__ = [
    # Schemas & validation
    "chain_schema",
    "model_schema",
    "method_schema",
    "pose_schema",
    "problem_schema",
    "validate_problem",
    "validate_chain_model",
    "validate_high_level_model",
    # Loading
    "load_problem_from_file",
    "load_json_or_yaml",
    # Builders
    "build_chain_from_model",
    "build_chain_from_low_level",
    "build_chain_from_high_level",
    # Saving
    "save_solutions_json",
    "save_problem_json",
]
