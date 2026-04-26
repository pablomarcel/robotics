# motion/utils.py
"""
Utility helpers for the Motion kinematics toolkit.

Highlights
----------
- @timing decorator that records elapsed_ms on dict returns (and as attribute otherwise)
- Filesystem helpers (ensure_dir)
- JSON helpers (NumPy- and dataclass-friendly)
- Numeric validators (is_rotation_matrix, is_se3)
- Angle helpers (to_radians, to_degrees)
- Small array helpers (as_vec3, almost_equal)

These are intentionally dependency-light for easy testing.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import functools
import json
import os
import time

import numpy as np


# ----------------------------- decorators ------------------------------------

def timing(fn: Callable) -> Callable:
    """
    Decorator that measures wall-clock time for a call.

    Behavior
    --------
    - If the function returns a dict, injects 'elapsed_ms' (float).
    - Otherwise, attaches an attribute '_elapsed_ms' to the return value.

    Examples
    --------
    @timing
    def compute(...): return {"value": 42}

    out = compute(...)
    assert "elapsed_ms" in out
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        if isinstance(out, dict):
            out.setdefault("elapsed_ms", dt)
        else:
            try:
                setattr(out, "_elapsed_ms", dt)
            except Exception:
                pass
        return out
    return wrapper


# ------------------------------ filesystem -----------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create the directory (and parents) if it doesn't exist and return it as Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------ JSON helpers ---------------------------------

class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that understands NumPy arrays/scalars and dataclasses.
    """
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        return super().default(obj)


def to_json(obj: Any, path: Union[str, Path]) -> str:
    """
    Serialize `obj` to JSON at `path` using NumpyJSONEncoder.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, cls=NumpyJSONEncoder, indent=2)
    return str(path)


def from_json(path: Union[str, Path]) -> Any:
    """
    Load JSON from `path`.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------ angle helpers --------------------------------

def to_radians(angle: float, *, degrees: bool = False) -> float:
    """
    Convert `angle` to radians if `degrees=True`, otherwise pass through.
    """
    return float(np.deg2rad(angle)) if degrees else float(angle)


def to_degrees(angle: float) -> float:
    """
    Convert radians to degrees.
    """
    return float(np.rad2deg(angle))


# ------------------------------ array helpers --------------------------------

def as_vec3(v: Iterable[float]) -> np.ndarray:
    """
    Coerce an iterable into a shape-(3,) float array; raises on bad input.
    """
    a = np.asarray(list(v), dtype=float).reshape(-1)
    if a.size != 3 or not np.all(np.isfinite(a)):
        raise ValueError("Expected a 3-vector of finite numbers")
    return a


def almost_equal(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Convenience wrapper around np.allclose for arrays/matrices.
    """
    return bool(np.allclose(a, b, atol=atol))


# --------------------------- numeric validators -------------------------------

def is_rotation_matrix(R: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Check if a 3x3 matrix is a proper rotation_kinematics:
      RᵀR = I  and det(R) = +1
    """
    R = np.asarray(R, float)
    if R.shape != (3, 3):
        return False
    I = np.eye(3)
    ortho = np.allclose(R.T @ R, I, atol=atol)
    det1 = np.isclose(np.linalg.det(R), 1.0, atol=atol)
    return bool(ortho and det1)


def is_se3(T: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Check if a 4x4 matrix is a valid homogeneous transform:
      top-left 3x3 is rotation_kinematics and last row is [0 0 0 1]
    """
    T = np.asarray(T, float)
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3], np.array([0, 0, 0, 1.0]), atol=atol):
        return False
    return is_rotation_matrix(T[:3, :3], atol=atol)


# ------------------------------- misc ----------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp a scalar to the inclusive range [lo, hi].
    """
    return float(min(max(x, lo), hi))


def version_string() -> str:
    """
    Basic version string. Bump here when API changes materially.
    """
    return "0.1.0"
