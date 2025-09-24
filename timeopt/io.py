# time/io.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from .utils import ensure_dir, save_json

def default_out_dir() -> Path:
    return ensure_dir("time/out")

def write_result_payload(name: str, data: Dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    ensure_dir(out)
    return save_json(out / f"{name}.json", data)
