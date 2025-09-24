from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, yaml

ROOT = Path(__file__).resolve().parent
IN  = ROOT / "in"
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True); IN.mkdir(parents=True, exist_ok=True)

@dataclass(slots=True)
class JsonStore:
    def write(self, name: str, obj):
        p = OUT / f"{name}.json"; p.write_text(json.dumps(obj, indent=2), encoding="utf-8"); return p
    def read(self, name: str): return json.loads((IN / f"{name}.json").read_text(encoding="utf-8"))

@dataclass(slots=True)
class YamlStore:
    def write(self, name: str, obj):
        p = OUT / f"{name}.yaml"; p.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8"); return p
    def read(self, name: str): return yaml.safe_load((IN / f"{name}.yaml").read_text(encoding="utf-8"))
