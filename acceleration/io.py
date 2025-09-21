# acceleration/io.py
"""
I/O, schema, and builders for **acceleration-kinematics** problems.

Problem shape
-------------
Acceleration problems are expressed as a small envelope:

  {
    "op": "forward" | "inverse" | "classic" | "euler_alpha" | "quat_sb" | "mixed",
    "payload": { ... op-specific fields ... },
    "model":   { "kind": "planar2r", "l1": <float>, "l2": <float> }  # required for forward/inverse
  }

Notes
-----
* Unlike the inverse module, acceleration workflows do **not** require low-level
  DH/MDH chain specs here. We build `ChainKinematics` façades from high-level models.
* YAML input is supported if PyYAML is installed.
* Validation uses `jsonschema` if available; otherwise we raise clear errors.

Examples
--------
Forward acceleration (planar 2R):
{
  "op": "forward",
  "model":   {"kind": "planar2r", "l1": 1.0, "l2": 0.7},
  "payload": {"q": [0.1, 0.2], "qd": [0.3, -0.1], "qdd": [0.0, 1.2]}
}

Inverse acceleration:
{
  "op": "inverse",
  "model":   {"kind": "planar2r", "l1": 1.0, "l2": 0.7},
  "payload": {"q": [0.1, 0.2], "qd": [0.3, -0.1], "xdd": [0.2, -0.5]}
}

Classic point acceleration:
{
  "op": "classic",
  "payload": {"alpha": [0,0,1], "omega": [0,0,3], "r": [0.2,0.0,0.0]}
}

Mixed (representative §9.4xx helper):
{
  "op": "mixed",
  "payload": {
    "R": [[1,0,0],[0,1,0],[0,0,1]],
    "omega": [0,0,2],
    "alpha": [0,0,1],
    "r": [0.1,0.0,0.0],
    "vB": [0.0,0.05,0.0]
  }
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import json
import numpy as np

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:  # pragma: no cover
    yaml = None
    _HAVE_YAML = False

from .core import ChainKinematics  # façade type
from . import design as design_mod  # presets

PathLike = Union[str, Path]


# -----------------------------------------------------------------------------
# JSON Schema helpers
# -----------------------------------------------------------------------------

def _number() -> Dict[str, Any]:
    return {"type": "number"}


def _vec_n(n: int) -> Dict[str, Any]:
    return {"type": "array", "items": _number(), "minItems": n, "maxItems": n}


def _mat_3x3() -> Dict[str, Any]:
    return {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": _number(),
        },
    }


# -----------------------------------------------------------------------------
# Schemas: model & per-op payloads
# -----------------------------------------------------------------------------

def model_schema() -> Dict[str, Any]:
    """Schema for **high-level** model specs used by acceleration presets."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Acceleration — High-level Model",
        "type": "object",
        "required": ["kind"],
        "properties": {
            "kind": {"type": "string", "enum": ["planar2r"]},
            "l1": _number(),
            "l2": _number(),
            "name": {"type": "string"},
        },
        "additionalProperties": True,  # allow forward-compatible fields
    }


def payload_forward_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["q", "qd", "qdd"],
        "properties": {
            "q": _vec_n(2),    # len not strictly enforced vs model.n; 2R default
            "qd": _vec_n(2),
            "qdd": _vec_n(2),
        },
        "additionalProperties": False,
    }


def payload_inverse_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["q", "qd", "xdd"],
        "properties": {
            "q": _vec_n(2),
            "qd": _vec_n(2),
            "xdd": {"type": "array", "items": _number(), "minItems": 2, "maxItems": 3},  # XY (or XYZ) style
            "damping": _number(),
        },
        "additionalProperties": False,
    }


def payload_classic_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["alpha", "omega", "r"],
        "properties": {"alpha": _vec_n(3), "omega": _vec_n(3), "r": _vec_n(3)},
        "additionalProperties": False,
    }


def payload_euler_alpha_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["angles", "rates", "accels"],
        "properties": {"angles": _vec_n(3), "rates": _vec_n(3), "accels": _vec_n(3)},
        "additionalProperties": False,
    }


def payload_quat_sb_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["q", "qd", "qdd"],
        "properties": {"q": _vec_n(4), "qd": _vec_n(4), "qdd": _vec_n(4)},
        "additionalProperties": False,
    }


def payload_mixed_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["R", "omega", "alpha", "r", "vB"],
        "properties": {
            "R": _mat_3x3(),
            "omega": _vec_n(3),
            "alpha": _vec_n(3),
            "r": _vec_n(3),
            "vB": _vec_n(3),
        },
        "additionalProperties": False,
    }


def problem_schema() -> Dict[str, Any]:
    """
    Top-level acceleration problem schema.

    Uses `if`/`then` to require `model` for forward/inverse ops, while keeping it
    optional for other ops.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Acceleration Problem",
        "type": "object",
        "required": ["op", "payload"],
        "properties": {
            "op": {"type": "string", "enum": [
                "forward", "inverse", "classic", "euler_alpha", "quat_sb", "mixed"
            ]},
            "payload": {"type": "object"},  # refined by oneOf below
            "model": model_schema(),
        },
        "allOf": [
            {
                "if": {"properties": {"op": {"const": "forward"}}, "required": ["op"]},
                "then": {"required": ["model"], "properties": {"payload": payload_forward_schema()}},
            },
            {
                "if": {"properties": {"op": {"const": "inverse"}}, "required": ["op"]},
                "then": {"required": ["model"], "properties": {"payload": payload_inverse_schema()}},
            },
            {
                "if": {"properties": {"op": {"const": "classic"}}, "required": ["op"]},
                "then": {"properties": {"payload": payload_classic_schema()}},
            },
            {
                "if": {"properties": {"op": {"const": "euler_alpha"}}, "required": ["op"]},
                "then": {"properties": {"payload": payload_euler_alpha_schema()}},
            },
            {
                "if": {"properties": {"op": {"const": "quat_sb"}}, "required": ["op"]},
                "then": {"properties": {"payload": payload_quat_sb_schema()}},
            },
            {
                "if": {"properties": {"op": {"const": "mixed"}}, "required": ["op"]},
                "then": {"properties": {"payload": payload_mixed_schema()}},
            },
        ],
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
    """Validate an acceleration problem against :func:`problem_schema`."""
    _validate(problem, problem_schema())


def validate_model(model: Mapping[str, Any]) -> None:
    """Validate a high-level acceleration model against :func:`model_schema`."""
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
    """Load an acceleration problem from JSON/YAML file."""
    data = load_json_or_yaml(path)
    if validate:
        validate_problem(data)
    return data


# -----------------------------------------------------------------------------
# Builders (high-level → ChainKinematics)
# -----------------------------------------------------------------------------

def build_chain_from_model(model: Mapping[str, Any], *, validate: bool = False) -> ChainKinematics:
    """
    Build a :class:`ChainKinematics` from a **high-level** model dict.
    Currently supported kinds:
      - {"kind": "planar2r", "l1": ..., "l2": ...}
    """
    if validate:
        validate_model(model)
    kind = str(model.get("kind", "")).strip().lower()
    if kind == "planar2r":
        l1 = float(model["l1"])
        l2 = float(model["l2"])
        return design_mod.planar_2r(l1, l2, name=str(model.get("name", "planar_2R")))
    raise ValueError(f"Unsupported model kind for acceleration: {kind!r}")


# -----------------------------------------------------------------------------
# Saving utilities
# -----------------------------------------------------------------------------

def save_result_json(path: PathLike, result: Any) -> None:
    """Save a generic result as pretty JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")


def save_problem_json(path: PathLike, problem: Mapping[str, Any]) -> None:
    """Save a problem dict as pretty JSON (without validation)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(problem), indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    # Schemas & validation
    "model_schema",
    "payload_forward_schema",
    "payload_inverse_schema",
    "payload_classic_schema",
    "payload_euler_alpha_schema",
    "payload_quat_sb_schema",
    "payload_mixed_schema",
    "problem_schema",
    "validate_problem",
    "validate_model",
    # Loading
    "load_problem_from_file",
    "load_json_or_yaml",
    # Builders
    "build_chain_from_model",
    # Saving
    "save_result_json",
    "save_problem_json",
]
