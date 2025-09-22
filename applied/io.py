# applied/io.py
"""
I/O, schema, and builders for **applied-dynamics** problems.

Two common actions are supported at the file level:

1) **Derive** (symbolic):
   {
     "action": "derive",
     "model": "simple_pendulum" | "spherical_pendulum" | "planar2r" | "cart_absorber",
     "params": { ... model-specific symbolic/numeric parameters ... }
   }

2) **Simulate** (numeric integration):
   {
     "action": "simulate",
     "model": "<same as above>",
     "params": { ... numeric parameters ... },
     "t_span": [t0, tf],
     "x0":     [initial_state...],
     "options": { "method": "RK45", "rtol": 1e-6, "atol": 1e-9, "max_step": null }
   }

Features
--------
- JSON **and** YAML input (auto-detected by extension).
- JSON Schema validation for model specs and derive/simulate requests.
- Builders for :class:`applied.core.System` via :mod:`applied.design.DesignLibrary`.
- Helpers to save/load problem payloads and results.

Examples
--------
Derive (Planar 2R, symbolic):
{
  "action": "derive",
  "model": "planar2r",
  "params": {"m1": "m1", "m2": "m2", "l1": "l1", "l2": "l2", "g": "g"}
}

Simulate (Simple pendulum, numeric):
{
  "action": "simulate",
  "model": "simple_pendulum",
  "params": {"m": 1.0, "l": 1.0, "g": 9.81},
  "t_span": [0.0, 10.0],
  "x0": [0.2, 0.0],  # [theta, theta_dot]
  "options": {"method": "RK45", "rtol": 1e-6, "atol": 1e-9}
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import json
import sympy as sp

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:  # pragma: no cover
    yaml = None
    _HAVE_YAML = False

# Keep imports minimal to avoid heavy load; design is light and avoids cycles
from . import design as design_mod
from .core import System

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# JSON Schemas
# ---------------------------------------------------------------------------

def _number() -> Dict[str, Any]:
    return {"type": "number"}


def _string() -> Dict[str, Any]:
    return {"type": "string"}


def _number_or_string() -> Dict[str, Any]:
    return {"oneOf": [_number(), _string()]}


def _array_numbers(min_items: int, max_items: Optional[int] = None) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "array",
        "items": _number(),
        "minItems": min_items,
    }
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def model_schema() -> Dict[str, Any]:
    """
    Schema for **high-level applied-dynamics model specs**.
    We accept numerics or symbol-names (strings) for parameters.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Applied dynamics — Model",
        "type": "object",
        "required": ["model", "params"],
        "properties": {
            "model": {
                "type": "string",
                "enum": [
                    "simple_pendulum",
                    "spherical_pendulum",
                    "planar2r",
                    "cart_absorber",
                ],
            },
            "params": {
                "type": "object",
                "additionalProperties": _number_or_string(),
                "properties": {
                    # simple_pendulum / spherical_pendulum
                    "m": _number_or_string(),
                    "l": _number_or_string(),
                    "g": _number_or_string(),
                    # planar2r
                    "m1": _number_or_string(),
                    "m2": _number_or_string(),
                    "l1": _number_or_string(),
                    "l2": _number_or_string(),
                    # cart_absorber
                    "M": _number_or_string(),
                    "k": _number_or_string(),
                },
            },
        },
        "additionalProperties": False,
    }


def derive_schema() -> Dict[str, Any]:
    """Schema for **derive** (symbolic) requests."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Applied dynamics — Derive request",
        "type": "object",
        "required": ["action", "model", "params"],
        "properties": {
            "action": {"type": "string", "const": "derive"},
            # Inline model spec (flattened at top level for convenience)
            "model": model_schema()["properties"]["model"],
            "params": model_schema()["properties"]["params"],
        },
        "additionalProperties": False,
    }


def simulate_schema() -> Dict[str, Any]:
    """Schema for **simulate** (numeric integration) requests."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Applied dynamics — Simulate request",
        "type": "object",
        "required": ["action", "model", "params", "t_span", "x0"],
        "properties": {
            "action": {"type": "string", "const": "simulate"},
            "model": model_schema()["properties"]["model"],
            "params": model_schema()["properties"]["params"],
            "t_span": _array_numbers(2, 2),
            "x0": {"type": "array", "items": _number(), "minItems": 1},
            "options": {
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "rtol": _number(),
                    "atol": _number(),
                    "max_step": {"oneOf": [{"type": "number"}, {"type": "null"}]},
                    "dt": {"oneOf": [{"type": "number"}, {"type": "null"}]},
                },
                "additionalProperties": True,
            },
        },
        "additionalProperties": False,
    }


def problem_schema() -> Dict[str, Any]:
    """
    Top-level schema: either a derive or simulate request.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Applied dynamics — Problem",
        "oneOf": [derive_schema(), simulate_schema()],
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate(instance: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "jsonschema is required for validation. Install with `pip install jsonschema`."
        ) from exc
    jsonschema.validate(instance=dict(instance), schema=dict(schema))


def validate_problem(problem: Mapping[str, Any]) -> None:
    _validate(problem, problem_schema())


def validate_model(model: str, params: Mapping[str, Any]) -> None:
    _validate({"model": model, "params": dict(params)}, model_schema())


def validate_derive_payload(payload: Mapping[str, Any]) -> None:
    _validate(payload, derive_schema())


def validate_simulate_payload(payload: Mapping[str, Any]) -> None:
    _validate(payload, simulate_schema())


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------

def _is_yaml_path(path: PathLike) -> bool:
    return Path(path).suffix.lower() in {".yaml", ".yml"}


def load_json_or_yaml(path: PathLike) -> Dict[str, Any]:
    """
    Load dict from a JSON or YAML file based on extension.
    """
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
    """
    Load an applied-dynamics problem (derive or simulate) from JSON/YAML.
    """
    data = load_json_or_yaml(path)
    if validate:
        validate_problem(data)
    return data


def save_problem_json(path: PathLike, payload: Mapping[str, Any]) -> None:
    """
    Save a derive/simulate payload to JSON (no validation enforced here).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def save_derivation_json(path: PathLike, result: Mapping[str, Any]) -> None:
    """
    Save a symbolic derivation result as JSON (stringified SymPy is fine).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(result), indent=2, default=str), encoding="utf-8")


def save_simulation_json(path: PathLike, t: Sequence[float], x: Sequence[Sequence[float]], **meta: Any) -> None:
    """
    Save a numeric solution (t, x) to JSON, with optional metadata (events, options...).
    """
    payload: Dict[str, Any] = {"t": list(map(float, t)), "x": [list(map(float, row)) for row in x]}
    if meta:
        # Basic JSON-serializable coercion
        def _to_jsonable(v):
            try:
                import numpy as _np  # type: ignore
                if isinstance(v, _np.ndarray):
                    return v.tolist()
            except Exception:
                pass
            if isinstance(v, (list, tuple)):
                return [_to_jsonable(x) for x in v]
            if isinstance(v, dict):
                return {k: _to_jsonable(val) for k, val in v.items()}
            return v
        payload.update({k: _to_jsonable(v) for k, v in meta.items()})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _parse_param_value(v: Any) -> Any:
    """
    Coerce parameter values:
      * numbers → numbers
      * strings → SymPy symbols (by name)
      * else    → passed through as-is
    """
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        # Simple symbol name; avoid advanced parsing for safety.
        return sp.Symbol(v)
    return v


def _parse_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: _parse_param_value(v) for k, v in params.items()}


def build_system_from_model(model: str, params: Mapping[str, Any]) -> System:
    """
    Build a :class:`applied.core.System` via :class:`applied.design.DesignLibrary`.
    """
    validate_model(model, params)
    lib = design_mod.DesignLibrary()
    return lib.create(model, **_parse_params(params))


# ---------------------------------------------------------------------------
# High-level helpers (optional sugar)
# ---------------------------------------------------------------------------

def is_derive(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("action", "")).lower() == "derive"


def is_simulate(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("action", "")).lower() == "simulate"


def pick_action(payload: Mapping[str, Any]) -> str:
    if is_derive(payload):
        return "derive"
    if is_simulate(payload):
        return "simulate"
    raise ValueError("payload.action must be 'derive' or 'simulate'.")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Schemas & validation
    "model_schema",
    "derive_schema",
    "simulate_schema",
    "problem_schema",
    "validate_problem",
    "validate_model",
    "validate_derive_payload",
    "validate_simulate_payload",
    # Loading/saving
    "load_problem_from_file",
    "load_json_or_yaml",
    "save_problem_json",
    "save_derivation_json",
    "save_simulation_json",
    # Builders
    "build_system_from_model",
    # Sugar
    "is_derive",
    "is_simulate",
    "pick_action",
]
