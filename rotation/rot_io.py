# =============================
# File: rotation/rot_io.py
# =============================
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np

from .rot_utils import ensure_dir, parse_floats


def read_points_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def write_points_csv(path: str, arr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.savetxt(path, arr, delimiter=",", fmt="%.9f")


def read_job(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith('.json'):
            return json.load(f)
        # simple key=value lines
        out: Dict[str, Any] = {}
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            if '=' in line:
                k,v = line.split('=',1)
                out[k.strip()] = v.strip()
        return out


def parse_angles_csv(s: str) -> List[float]:
    return parse_floats(s)