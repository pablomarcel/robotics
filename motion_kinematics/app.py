# motion_kinematics/app.py
"""
Top-level application orchestrator for Motion Kinematics.

This module provides a small facade (`App`) around the public APIs exposed by
`apis.py`. It is intentionally slim and side-effect free to make unit testing
straightforward while still being convenient for production use.

Usage (programmatic):
    from motion_kinematics.app import App, Config
    app = App()  # default config uses ./motion_kinematics/in and ./motion_kinematics/out
    T = app.screw(u=(0, 0, 1), s=(0, 0, 0), h=0.1, phi=1.0)

Usage (CLI):
    python -m motion_kinematics.cli  # (app.py delegates to cli.py when run as __main__)

Design notes:
- Pure OOP: the `App` manages configuration, logging, directory hygiene, and
  delegates all math/IO to the `APIs` facade (see apis.py).
- Testability: every public method returns data (no implicit printing), and any
  file output is optional and explicit via `output_path` or by using io.py.
- Extensibility: new high-level verbs should be thin pass-throughs to `APIs`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import json
import logging
import os
import sys

# --- Optional import of utilities (we fall back gracefully in early scaffolding)
try:  # pragma: no cover - exercised once modules exist
    from .utils import timing  # nice-to-have decorator (no-op fallback below)
except Exception:  # pragma: no cover
    def timing(fn):  # type: ignore
        return fn

try:  # pragma: no cover - exercised once modules exist
    from .apis import APIs
except Exception as exc:  # pragma: no cover
    # Provide a helpful error early in the piecemeal build.
    class _MissingAPIs:  # minimal sentinel to fail with guidance
        def __getattr__(self, name):
            raise RuntimeError(
                f"motion_kinematics.apis.APIs is not available yet. "
                f"Add apis.py next. Missing attribute: {name}"
            ) from exc

    APIs = _MissingAPIs  # type: ignore


# ----------------------------- Configuration ---------------------------------
@dataclass(frozen=True)
class Config:
    """
    Application configuration.

    Attributes
    ----------
    base_dir : Path
        Root directory of the motion_kinematics package (default: directory containing this file).
    in_dir : Path
        Directory where input files are read from (`motion_kinematics/in`).
    out_dir : Path
        Directory where output files are written (`motion_kinematics/out`).
    log_level : int
        Python logging level (default: logging.INFO).
    """
    base_dir: Path = Path(__file__).resolve().parent
    in_dir: Path = Path(__file__).resolve().parent / "in"
    out_dir: Path = Path(__file__).resolve().parent / "out"
    log_level: int = logging.INFO


# ------------------------------- App Facade ----------------------------------
class App:
    """
    High-level application facade.

    This class provides a thin, typed layer over the `APIs` facade, adds
    directory hygiene, and centralizes logging. It is safe to import and
    instantiate in unit tests.

    Parameters
    ----------
    config : Config, optional
        Application configuration. Defaults to `Config()` which uses
        `motion_kinematics/in` and `motion_kinematics/out`.
    apis : APIs, optional
        Dependency-injected facade for computations; defaults to `APIs()`.
    logger : logging.Logger, optional
        Inject a logger (useful for tests). Defaults to a configured logger.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        apis: Optional["APIs"] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or Config()
        self._ensure_dirs(self.config.in_dir, self.config.out_dir)
        self.logger = logger or self._build_logger(self.config.log_level)
        self.apis: "APIs" = apis or APIs()  # type: ignore[call-arg]

    # ---------------------------- Public API ----------------------------
    @timing
    def rotation(
        self,
        axis: Tuple[float, float, float],
        angle: float,
        *,
        degrees: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Build a 3x3 rotation_kinematics from axis-angle and (optionally) persist it.

        Returns
        -------
        dict
            Payload containing the rotation_kinematics matrix and metadata.
        """
        self.logger.debug("Building rotation_kinematics: axis=%s, angle=%s, degrees=%s", axis, angle, degrees)
        payload = self.apis.rotation_axis_angle(axis=axis, angle=angle, degrees=degrees)
        return self._maybe_write(payload, output_path)

    @timing
    def screw(
        self,
        u: Tuple[float, float, float],
        s: Tuple[float, float, float],
        h: float,
        phi: float,
        *,
        degrees: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Build an SE(3) transform from screw parameters (eqs. 4.206, 4.220–4.222).
        """
        self.logger.debug("Building screw: u=%s, s=%s, h=%s, phi=%s, degrees=%s", u, s, h, phi, degrees)
        payload = self.apis.screw_motion(u=u, s=s, h=h, phi=phi, degrees=degrees)
        return self._maybe_write(payload, output_path)

    @timing
    def plucker_line_from_points(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        *,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Create a Plücker line from two points (eqs. 4.344–4.346).
        """
        self.logger.debug("Building Plücker line from points: p1=%s, p2=%s", p1, p2)
        payload = self.apis.plucker_from_points(p1=p1, p2=p2)
        return self._maybe_write(payload, output_path)

    @timing
    def plucker_angle_distance(
        self,
        a1: Tuple[float, float, float],
        a2: Tuple[float, float, float],
        b1: Tuple[float, float, float],
        b2: Tuple[float, float, float],
        *,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Angle (4.389) and shortest distance (4.390) between two lines.
        """
        self.logger.debug("Plücker angle & distance: a1=%s, a2=%s, b1=%s, b2=%s", a1, a2, b1, b2)
        payload = self.apis.plucker_angle_distance(a1=a1, a2=a2, b1=b1, b2=b2)
        return self._maybe_write(payload, output_path)

    @timing
    def plane_point_distance(
        self,
        point: Tuple[float, float, float],
        normal: Tuple[float, float, float],
        *,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Distance of a point to a plane (cf. 4.391–4.397).
        """
        self.logger.debug("Plane-point distance: point=%s, normal=%s", point, normal)
        payload = self.apis.plane_point_distance(point=point, normal=normal)
        return self._maybe_write(payload, output_path)

    @timing
    def forward_kinematics(
        self,
        dh_params: Iterable[Iterable[float]],
        *,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Compute forward_kinematics kinematics from a DH table.
        Each row: [a, alpha, d, theta] (radians).
        """
        self.logger.debug("Forward kinematics with DH params (first row if any): %s",
                          next(iter(dh_params), "[]"))
        payload = self.apis.forward_kinematics(dh_params=dh_params)
        return self._maybe_write(payload, output_path)

    # ------------------------- Internals / Utilities --------------------
    @staticmethod
    def _ensure_dirs(*paths: Path) -> None:
        for p in paths:
            p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_logger(level: int) -> logging.Logger:
        logger = logging.getLogger("motion_kinematics.app")
        if not logger.handlers:
            handler = logging.StreamHandler(stream=sys.stdout)
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    def _maybe_write(
        self,
        payload: Dict[str, Any],
        output_path: Optional[Union[str, Path]],
    ) -> Dict[str, Any]:
        """
        Optionally write a JSON payload to `motion_kinematics/out` (or custom path).
        """
        if output_path is None:
            return payload

        path = Path(output_path)
        if not path.is_absolute():
            path = self.config.out_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, cls=_NumpyJSONEncoder, indent=2)
        self.logger.info("Wrote: %s", path)
        return payload


# --------------------------- JSON helper (local) -----------------------------
class _NumpyJSONEncoder(json.JSONEncoder):
    """A tiny encoder to safely dump numpy types/arrays if callers pass them."""
    def default(self, obj: Any) -> Any:  # pragma: no cover - behavior exercised via APIs
        try:
            import numpy as np  # local import to keep app.py lean at import time

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except Exception:
            pass
        return super().default(obj)


# --------------------------------- Main --------------------------------------
def main() -> None:
    """
    Delegate to the package CLI when app.py is executed directly.

    Having the delegation here makes `python -m motion_kinematics.app` work for folks who
    discover this module first.
    """
    try:
        # Defer import—CLI may pull in heavier deps (argparse, numpy printing).
        from .cli import main as cli_main  # type: ignore
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(
            "The CLI is not available yet (missing motion_kinematics/cli.py). "
            "Either add cli.py or use App() programmatically.\n"
        )
        raise
    cli_main()


if __name__ == "__main__":  # pragma: no cover
    main()
