"""
orientation.io (OO version)
---------------------------
Object-oriented I/O utilities for the orientation package.

Design
------
- PathConfig: holds the canonical input/output directory layout.
- IOManager: high-level read/write facade (JSON and CSV matrices).
- A default singleton `IO` is exported, along with thin shims `read_json`,
  `write_json`, `write_matrix_csv` etc., for convenience and compatibility.

Security
--------
- All writes create parent directories as needed.
- Filenames are validated to remain under the configured in/out dirs
  (prevents path traversal).

Author: Robotics project
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import csv
import json
import numpy as np


# --------------------------------------------------------------------------
# Path configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PathConfig:
    """Directory layout for the app.

    Parameters
    ----------
    base_dir : Path
        The directory containing the `orientation` package files.
    in_subdir : str
        Subdirectory (relative to base_dir) for inputs.
    out_subdir : str
        Subdirectory (relative to base_dir) for outputs.
    """
    base_dir: Path
    in_subdir: str = "in"
    out_subdir: str = "out"

    @property
    def in_dir(self) -> Path:
        return (self.base_dir / self.in_subdir).resolve()

    @property
    def out_dir(self) -> Path:
        return (self.base_dir / self.out_subdir).resolve()


# --------------------------------------------------------------------------
# IO manager
# --------------------------------------------------------------------------

class IOManager:
    """High-level, testable I/O facade for JSON and CSV matrices."""

    def __init__(self, config: Optional[PathConfig] = None):
        if config is None:
            # default: directories alongside this file
            pkg_dir = Path(__file__).resolve().parent
            config = PathConfig(base_dir=pkg_dir)
        self.config = config
        # Ensure directories exist at construction time (idempotent)
        self.config.in_dir.mkdir(parents=True, exist_ok=True)
        self.config.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- internal helpers ----------------

    def _safe_path(self, rel_or_name: str | Path, *, kind: str) -> Path:
        """Return a path inside in_dir/out_dir; raises on traversal."""
        base = self.config.in_dir if kind == "in" else self.config.out_dir
        p = (base / Path(rel_or_name)).resolve()
        if not str(p).startswith(str(base)):
            raise ValueError(f"Path escapes {kind} dir: {p}")
        return p

    # ---------------- JSON ----------------

    def read_json(self, name: str) -> dict:
        """Read JSON from `in/` if present; otherwise fall back to `out/`."""
        try_first = self._safe_path(name, kind="in")
        path = try_first if try_first.exists() else self._safe_path(name, kind="out")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, name: str, data: dict) -> None:
        """Write JSON to `out/` by filename."""
        path = self._safe_path(name, kind="out")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---------------- CSV matrices ----------------

    def write_matrix_csv(self, name: str, M: np.ndarray) -> None:
        """Write a 2D numpy array as CSV into `out/`."""
        path = self._safe_path(name, kind="out")
        path.parent.mkdir(parents=True, exist_ok=True)
        M = np.asarray(M, dtype=float)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in M.tolist():
                w.writerow(row)

    def read_matrix_csv(self, name: str) -> np.ndarray:
        """Read a CSV from `in/` or `out/` (prefers `in/`), returning np.ndarray."""
        try_first = self._safe_path(name, kind="in")
        if try_first.exists():
            target = try_first
        else:
            target = self._safe_path(name, kind="out")
        with target.open("r", newline="", encoding="utf-8") as f:
            rows: List[List[float]] = []
            for r in csv.reader(f):
                if not r:
                    continue
                rows.append([float(x) for x in r])
        return np.array(rows, dtype=float)

    # ---------------- convenience for batch jobs ----------------

    def read_jobs(self, name: str) -> list[dict]:
        """Read a batch jobs file (JSON list) from `in/`."""
        data = self.read_json(name)
        if isinstance(data, list):
            return data
        # Allow {"jobs": [...]} too
        return data.get("jobs", [])

    def write_results(self, name: str, results: list[dict]) -> None:
        """Write batch results JSON to `out/`."""
        self.write_json(name, {"results": results})


# --------------------------------------------------------------------------
# Default singleton + compatibility shims
# --------------------------------------------------------------------------

# Default IO manager rooted at the package directory
IO = IOManager()

# Expose canonical dirs (used by CLI help/printing)
IN_DIR: Path = IO.config.in_dir
OUT_DIR: Path = IO.config.out_dir

# Thin functional shims for convenience/back-compat with earlier examples
def read_json(name: str) -> dict:
    return IO.read_json(name)

def write_json(name: str, data: dict) -> None:
    IO.write_json(name, data)

def write_matrix_csv(name: str, M: np.ndarray) -> None:
    IO.write_matrix_csv(name, M)

def read_matrix_csv(name: str) -> np.ndarray:
    return IO.read_matrix_csv(name)
