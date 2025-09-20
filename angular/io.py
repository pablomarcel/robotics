from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Any, Dict, Tuple
from .utils import save_array, load_array

IN_DIR = Path(__file__).parent / "in"
OUT_DIR = Path(__file__).parent / "out"
IN_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)

@dataclass
class IOManager:
    """File I/O centered on angular/in and angular/out."""
    in_dir: Path = IN_DIR
    out_dir: Path = OUT_DIR

    def read_json(self, name: str) -> Dict[str, Any]:
        return json.loads((self.in_dir / name).read_text())

    def write_json(self, name: str, data: Dict[str, Any]) -> None:
        (self.out_dir / name).write_text(json.dumps(data, indent=2))

    def save_npy(self, name: str, arr: np.ndarray) -> None:
        save_array(str(self.out_dir / name), arr)

    def load_npy(self, name: str) -> np.ndarray:
        return load_array(str(self.in_dir / name))
