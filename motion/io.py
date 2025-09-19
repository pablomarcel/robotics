# motion/io.py
"""
I/O utilities for the Motion Kinematics toolkit.

Design goals
------------
- Pure OOP service (`IO`) with explicit config (dataclass).
- Deterministic, atomic writes (tmp → rename) for robustness.
- Minimal deps (stdlib + numpy).
- Friendly helpers for common artifacts:
  • JSON payloads (dicts / dataclasses / numpy arrays)
  • Matrices (.npy), point clouds (.csv), DH tables (.csv/.json)
  • SE(3) transforms (both .npy and .json flavors)
- All relative paths are resolved under motion/{in,out}.

Typical usage
-------------
from motion.io import IO, IOConfig

io = IO()  # defaults to motion/in and motion/out (relative to this package)
path = io.save_json({"hello": "world"}, "example.json")
T = np.eye(4); io.save_transform(T, "pose_A")  # writes pose_A.npy and pose_A.json
P = io.load_points_csv("cloud.csv")            # reads motion/in/cloud.csv by default
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import csv
import io as _io
import json
import os
import tempfile

import numpy as np


# ------------------------------- config --------------------------------------

@dataclass(frozen=True)
class IOConfig:
    """
    Configuration for the IO service.

    Attributes
    ----------
    base_dir : Path
        Root of the motion package (defaults to directory of this file).
    in_dir : Path
        Directory for inputs (defaults to <base_dir>/in).
    out_dir : Path
        Directory for outputs (defaults to <base_dir>/out).
    """
    base_dir: Path = Path(__file__).resolve().parent
    in_dir: Path = Path(__file__).resolve().parent / "in"
    out_dir: Path = Path(__file__).resolve().parent / "out"


# ------------------------------ json encoder ---------------------------------

class _NumpyJSONEncoder(json.JSONEncoder):
    """Tiny encoder to make numpy and dataclasses serializable."""
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
                return obj.item()
        except Exception:
            pass
        return super().default(obj)


# ------------------------------- io service ----------------------------------

class IO:
    """
    High-level, side-effect controlled file I/O service.
    """

    def __init__(self, config: Optional[IOConfig] = None) -> None:
        self.cfg = config or IOConfig()
        self._ensure(self.cfg.in_dir)
        self._ensure(self.cfg.out_dir)

    # ------------------------------ helpers ------------------------------
    @staticmethod
    def _ensure(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def _resolve_in(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.cfg.in_dir / p)

    def _resolve_out(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.cfg.out_dir / p)

    @staticmethod
    def _atomic_write_bytes(data: bytes, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)

    @staticmethod
    def _atomic_write_text(text: str, path: Path, encoding: str = "utf-8") -> None:
        IO._atomic_write_bytes(text.encode(encoding), path)

    # ------------------------------- JSON --------------------------------
    def save_json(self, obj: Any, name_or_path: Union[str, Path]) -> str:
        """
        Save any JSON-serializable object (dataclasses & numpy supported).

        If `name_or_path` is relative, it is written under motion/out/.
        Returns the absolute path as a string.
        """
        path = self._resolve_out(name_or_path)
        payload = json.dumps(obj, cls=_NumpyJSONEncoder, indent=2)
        self._atomic_write_text(payload, path)
        return str(path)

    def load_json(self, name_or_path: Union[str, Path]) -> Any:
        """
        Load JSON from motion/in (if relative) or an absolute path.
        """
        path = self._resolve_in(name_or_path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------ matrices -----------------------------
    def save_matrix(self, M: np.ndarray, name_stem: str) -> str:
        """
        Save a numpy array to .npy under motion/out/.

        Example:
            save_matrix(T, "pose_A") -> motion/out/pose_A.npy
        """
        path = self._resolve_out(f"{name_stem}.npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use NamedTemporaryFile is tricky with np.save; write to BytesIO then atomic write.
        bio = _io.BytesIO()
        np.save(bio, M)
        self._atomic_write_bytes(bio.getvalue(), path)
        return str(path)

    def load_matrix(self, name_or_path: Union[str, Path]) -> np.ndarray:
        """
        Load a .npy array from motion/in (if relative) or absolute path.
        """
        path = self._resolve_in(name_or_path)
        return np.load(str(path))

    # ---------------------------- transforms -----------------------------
    def save_transform(self, T: np.ndarray, name_stem: str) -> Dict[str, str]:
        """
        Save an SE(3) 4x4 transform as both .npy and .json for convenience.

        Returns
        -------
        dict with keys: 'npy', 'json'
        """
        npy = self.save_matrix(T, f"{name_stem}")
        # Also split into R,t for a readable JSON view
        R = T[:3, :3]
        t = T[:3, 3]
        js = self.save_json({"R": R, "t": t, "T": T}, f"{name_stem}.json")
        return {"npy": npy, "json": js}

    def load_transform(self, name_or_path: Union[str, Path]) -> np.ndarray:
        """
        Load an SE(3) transform.

        If a JSON file is provided, we reconstruct T from R,t if present,
        else load the 'T' field. If a NPY file is provided, we return it directly.
        """
        path = Path(name_or_path)
        if not path.suffix:  # no extension given → prefer .npy under in/
            path = self._resolve_in(f"{path.name}.npy")
        if path.suffix.lower() == ".npy":
            return np.load(str(path))
        # assume JSON
        obj = self.load_json(path)
        if isinstance(obj, dict):
            if "T" in obj:
                return np.asarray(obj["T"], dtype=float)
            if "R" in obj and "t" in obj:
                T = np.eye(4)
                T[:3, :3] = np.asarray(obj["R"], float)
                T[:3, 3] = np.asarray(obj["t"], float).reshape(3)
                return T
        raise ValueError(f"Cannot reconstruct SE(3) from {path}")

    # ---------------------------- point clouds ---------------------------
    def save_points_csv(self, P: np.ndarray, name_or_path: Union[str, Path]) -> str:
        """
        Save Nx3 points as CSV (no header) under motion/out/.
        """
        path = self._resolve_out(name_or_path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False, newline="") as tmp:
            w = csv.writer(tmp)
            for row in np.asarray(P, float).reshape(-1, 3):
                w.writerow([f"{float(row[0])}", f"{float(row[1])}", f"{float(row[2])}"])
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        return str(path)

    def load_points_csv(self, name_or_path: Union[str, Path]) -> np.ndarray:
        """
        Load Nx3 points from CSV in motion/in/ (if relative) or absolute path.
        """
        path = self._resolve_in(name_or_path)
        rows: List[List[float]] = []
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            for r in reader:
                if not r:
                    continue
                rows.append([float(r[0]), float(r[1]), float(r[2])])
        return np.asarray(rows, dtype=float)

    # ----------------------------- DH tables -----------------------------
    def save_dh_csv(self, dh_rows: Iterable[Iterable[float]], name_or_path: Union[str, Path]) -> str:
        """
        Save a DH table (rows of [a, alpha, d, theta]) as CSV.
        """
        path = self._resolve_out(name_or_path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False, newline="") as tmp:
            w = csv.writer(tmp)
            for row in dh_rows:
                a, alpha, d, theta = [float(x) for x in row]
                w.writerow([a, alpha, d, theta])
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        return str(path)

    def load_dh_csv(self, name_or_path: Union[str, Path]) -> List[Tuple[float, float, float, float]]:
        """
        Load a DH table from CSV under motion/in/.
        """
        path = self._resolve_in(name_or_path)
        out: List[Tuple[float, float, float, float]] = []
        with path.open("r", newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                a, alpha, d, theta = [float(x) for x in row[:4]]
                out.append((a, alpha, d, theta))
        return out

    def save_dh_json(self, dh_rows: Iterable[Iterable[float]], name_or_path: Union[str, Path]) -> str:
        """
        Save a DH table as JSON (list of rows).
        """
        rows = [[float(x) for x in row] for row in dh_rows]
        return self.save_json(rows, name_or_path if str(name_or_path).endswith(".json") else f"{name_or_path}.json")

    def load_dh_json(self, name_or_path: Union[str, Path]) -> List[Tuple[float, float, float, float]]:
        """
        Load a DH table from JSON under motion/in/.
        """
        rows = self.load_json(name_or_path)
        return [(float(a), float(alpha), float(d), float(theta)) for a, alpha, d, theta in rows]

    # --------------------------- directory listings -----------------------
    def list_inputs(self, pattern: str = "*") -> List[str]:
        """
        List files under motion/in matching a glob pattern.
        """
        return sorted(str(p) for p in self.cfg.in_dir.glob(pattern))

    def list_outputs(self, pattern: str = "*") -> List[str]:
        """
        List files under motion/out matching a glob pattern.
        """
        return sorted(str(p) for p in self.cfg.out_dir.glob(pattern))
