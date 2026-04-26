# velocity_kinematics/app.py
"""
Velocity Kinematics Toolkit – App bootstrap.

Typical usage
-------------
python -m velocity_kinematics.app fk --robot_dynamics velocity_kinematics/in/planar2r.json --q 0.3,0.2
python -m velocity_kinematics.app jacobian --robot_dynamics velocity_kinematics/in/planar2r.json --q 0.3,0.2
python -m velocity_kinematics.app newton-ik --robot_dynamics velocity_kinematics/in/planar2r.json --q0 0.1,0.1 --p 1.0,0.5,0.0
python -m velocity_kinematics.app lu-solve --A '[[2,1],[1,3]]' --b '[1,2]'
python -m velocity_kinematics.app diagram --out velocity_kinematics/out

What this file does
-------------------
- Sets up lightweight logging (honors VELOCITY_LOGLEVEL).
- Ensures default input/output folders exist (velocity_kinematics/in, velocity_kinematics/out).
- Delegates all command handling to `velocity_kinematics.cli.VelocityCLI`.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .cli import VelocityCLI

__all__ = ["AppConfig", "setup_logging", "ensure_workdirs", "main", "__version__"]
__version__ = "0.1.0"


# ------------------------------- Configuration -------------------------------

@dataclass(frozen=True)
class AppConfig:
    """
    Simple configuration container for common paths.

    Attributes
    ----------
    root : Path
        Project root (directory that contains this file).
    indir : Path
        Input directory (default: velocity_kinematics/in).
    outdir : Path
        Output directory (default: velocity_kinematics/out).
    """
    root: Path
    indir: Path
    outdir: Path

    @staticmethod
    def default() -> "AppConfig":
        root = Path(__file__).resolve().parent
        return AppConfig(
            root=root,
            indir=root / "in",
            outdir=root / "out",
        )


# ------------------------------- Bootstrapping --------------------------------

def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure logging for the application.

    Parameters
    ----------
    level : Optional[str]
        Log level name ('INFO', 'DEBUG', ...). If None, uses VELOCITY_LOGLEVEL or 'INFO'.
    """
    lvl = (level or os.getenv("VELOCITY_LOGLEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger(__name__).debug("Logging initialized at level %s", lvl)


def ensure_workdirs(cfg: AppConfig) -> None:
    """
    Ensure input/output folders exist (create if missing).

    Mirrors project convention:
      - velocity_kinematics/in  : for input files (DH, matrices, etc.)
      - velocity_kinematics/out : for results (reports, diagrams, dumps)
    """
    cfg.indir.mkdir(parents=True, exist_ok=True)
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).debug("Workdirs ready: %s, %s", cfg.indir, cfg.outdir)


# --------------------------------- Entrypoint ---------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    App entrypoint (used by tests and `python -m velocity_kinematics.app`).

    Returns a process-style exit code from the CLI:
      0 = success
      1 = unexpected error (caught/handled inside CLI)
      2 = API/usage error (caught/handled inside CLI)
    """
    setup_logging()
    cfg = AppConfig.default()
    ensure_workdirs(cfg)
    # Delegate to the CLI facade; it handles parsing & dispatch internally.
    return VelocityCLI().run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
