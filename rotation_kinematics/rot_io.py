# =============================
# File: rotation_kinematics/rot_io.py
# =============================
from __future__ import annotations
import json
import os
from typing import Any, Dict

import numpy as np

from .rot_utils import ensure_dir, parse_floats

try:
    import yaml
    _YAML_OK = True
except Exception:
    _YAML_OK = False

def read_points_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")

def write_points_csv(path: str, arr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.savetxt(path, arr, delimiter=",", fmt="%.9f")

def read_job(path: str) -> Dict[str, Any]:
    """
    Read a batch job file:
      - .json : JSON
      - .yaml/.yml : YAML (requires PyYAML)
    Returns a dict with at least {"tasks": [...]}.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            return json.load(f)
        if ext in {".yaml", ".yml"}:
            if not _YAML_OK:
                raise RuntimeError("PyYAML not installed. pip install pyyaml")
            return yaml.safe_load(f)
        # fallback: simple key=value lines
        out: Dict[str, Any] = {}
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
        return out
