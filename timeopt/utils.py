# time/utils.py
from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import Any, Dict

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(p: str | Path, obj: Dict[str, Any]) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return p

def load_json(p: str | Path) -> Dict[str, Any]:
    from pathlib import Path
    return json.loads(Path(p).read_text(encoding="utf-8"))

def try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None
