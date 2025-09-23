from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

IN_DIR = Path(__file__).resolve().parent / "in"
OUT_DIR = Path(__file__).resolve().parent / "out"
IN_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)

@dataclass
class IOManager:
    """Tiny IO facade for reading/writing JSON/CSV arrays in path/in and path/out."""
    in_dir: Path = IN_DIR
    out_dir: Path = OUT_DIR

    def read_json(self, name: str) -> dict:
        return json.loads((self.in_dir / name).read_text())

    def write_json(self, name: str, payload: dict) -> Path:
        p = self.out_dir / name
        p.write_text(json.dumps(payload, indent=2))
        return p

    def write_csv(self, name: str, **arrays) -> Path:
        p = self.out_dir / name
        cols = list(arrays.keys())
        data = np.column_stack([np.asarray(arrays[k]).ravel() for k in cols])
        header = ",".join(cols)
        np.savetxt(p, data, delimiter=",", header=header, comments="")
        return p
