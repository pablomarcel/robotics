from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml
import json
import numpy as np
from .core import Link, Joint, RobotModel


@dataclass(slots=True)
class IOConfig:
    in_dir: Path
    out_dir: Path


class IOMgr:
    """Read/write robot_dynamics descriptions and numeric results.

    Input format: YAML with `links`, `joints`, optional `dh`.
    Output: JSON/CSV arrays for M, C, g, tau.
    """

    def __init__(self, cfg: IOConfig):
        self.cfg = cfg
        self.cfg.in_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

    def load_yaml(self, name: str) -> Dict[str, Any]:
        with open(self.cfg.in_dir / name, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def save_json(self, name: str, data: Dict[str, Any]) -> None:
        with open(self.cfg.out_dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def model_from_yaml(self, name: str) -> RobotModel:
        d = self.load_yaml(name)
        links = [
            Link(L["name"], float(L["mass"]), np.array(L["com"], float), np.array(L["inertia"], float), L.get("length"))
            for L in d["links"]
        ]
        joints = [Joint(J["name"], J["type"]) for J in d["joints"]]
        return RobotModel(d.get("name", "Robot"), links, joints)