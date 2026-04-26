# acceleration_kinematics/app.py
"""
High-level façade for the acceleration_kinematics-kinematics package.

This class mirrors the inverse_kinematics module’s app: it centralizes file I/O,
preset builders, forward_kinematics/inverse_kinematics acceleration_kinematics, and small convenience
APIs (classic acceleration_kinematics, Euler/Quaternion, mixed derivatives) behind
a small, testable surface. Importantly, it **does not** import
`acceleration_kinematics.apis` to avoid accidental circular imports with any CLI or
service layers. It uses the primitives from `acceleration_kinematics.core` and the
lightweight numpy backend so tests run with zero heavy deps.

Supported operations (extensible):
- forward_accel:  ẍ = J q̈ + J̇ q̇  (9.283)
- inverse_accel:  q̈ = J⁺ (ẍ − J̇ q̇) (9.291 / 9.327 damped)
- classic:        α×r + ω×(ω×r)     (tangential + centripetal)
- euler_alpha:    ZYX Euler angular_velocity acceleration_kinematics (9.127–9.131)
- quaternion_SB:  quaternion-based acceleration_kinematics transform (9.175–9.181)
- mixed:          representative mixed acceleration_kinematics cases (9.400–9.426)

Problem runner
--------------
`solve(problem_dict)` accepts a uniform payload and routes to the
requested operation. See docstrings for each method for the expected
shape.

Class diagram helpers
---------------------
`class_diagram_dot()` returns Graphviz DOT, and
`generate_class_diagram()` writes a Mermaid markdown diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union, get_args, get_origin

import importlib
import inspect
import json
import numpy as np

from .utils import timed
from .core import (
    ClassicAccel,
    MixedAcceleration,
    EulerKinematics,
    QuaternionKinematics,
    ChainKinematics,
)
from .backends.numpy_backend import Planar2R
from . import io as io_mod
from . import design as design_mod  # (placeholder; keeps symmetry with inverse_kinematics)
# If you later add design helpers (e.g., DH builders), we can route here.

# Optional: if you add tools.diagram like the inverse_kinematics module, wire it:
try:
    from .tools.diagram import render_dot as _render_dot  # type: ignore
except Exception:  # pragma: no cover - optional helper
    _render_dot = None  # graceful fallback

Number = Union[int, float]
ArrayLike = Union[Sequence[Number], np.ndarray]
PathLike = Union[str, Path]


# -----------------------------------------------------------------------------
# App metadata
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AppInfo:
    """Immutable metadata about the application."""
    name: str = "acceleration_kinematics-kinematics"
    version: str = "0.1.0"
    homepage: str = "https://example.local/acceleration"  # placeholder


# -----------------------------------------------------------------------------
# Acceleration App
# -----------------------------------------------------------------------------

class AccelApp:
    """
    High-level application façade for acceleration_kinematics-kinematics workflows.

    Responsibilities
    ----------------
    * Build robot_dynamics backends (Planar 2R out of the box; backends pluggable).
    * Forward & inverse_kinematics acceleration_kinematics for chains (9.283 / 9.291 / 9.327).
    * Classic tangential+centripetal term (α×r + ω×(ω×r)).
    * Euler/Quaternion helpers (orientation_kinematics kinematics).
    * Mixed accelerations (representative helpers for 9.400–9.426).
    * Problem-API utilities used by CLI/tests (solve / batch / from-file).
    * Class diagram export (DOT text) and Mermaid markdown emitter.
    """

    # ---------- lifecycle ----------

    def __init__(self, in_dir: Optional[PathLike] = None, out_dir: Optional[PathLike] = None):
        self.info = AppInfo()
        self.in_dir = Path(in_dir) if in_dir else Path("acceleration_kinematics/in")
        self.out_dir = Path(out_dir) if out_dir else Path("acceleration_kinematics/out")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Presets / backends ----------

    def preset_planar_2r(self, l1: Number, l2: Number) -> ChainKinematics:
        """
        Return a ChainKinematics façade over the numpy Planar2R backend.

        This mirrors the inverse_kinematics app’s `preset_planar_2r`, but here we return
        a ChainKinematics so we can call `forward_accel` / `inverse_accel`.
        """
        backend = Planar2R(float(l1), float(l2))
        return ChainKinematics(backend=backend, frame="ee")

    # In future you can add:
    # - preset_from_pinocchio(model_path) -> ChainKinematics
    # - preset_from_drake(plant_builder)  -> ChainKinematics

    # ---------- Chain acceleration_kinematics (forward_kinematics / inverse_kinematics) ----------

    @timed
    def forward_accel(
        self,
        chain: ChainKinematics,
        q: ArrayLike,
        qd: ArrayLike,
        qdd: ArrayLike,
    ):
        """
        Forward acceleration_kinematics (9.283):  ẍ = J q̈ + J̇ q̇

        Returns
        -------
        result: object with `.xdd` (np.ndarray) for convenience in tests.
        """
        xdd = chain.forward_accel(q, qd, qdd)

        class _Res:
            def __init__(self, x): self.xdd = np.asarray(x, float).reshape(-1)
        return _Res(xdd)

    @timed
    def inverse_accel(
        self,
        chain: ChainKinematics,
        q: ArrayLike,
        qd: ArrayLike,
        xdd: ArrayLike,
        *,
        damping: float = 1e-8,
    ):
        """
        Inverse acceleration_kinematics (9.291 / 9.327 damped): q̈ = J⁺ (ẍ − J̇ q̇)

        Parameters
        ----------
        damping : DLS damping (λ) for numerical robustness.
        """
        qdd = chain.inverse_accel(q, qd, xdd, damp=float(damping))

        class _Res:
            def __init__(self, qdd): self.qdd = np.asarray(qdd, float).reshape(-1)
        return _Res(qdd)

    # ---------- Classic / Euler / Quaternion / Mixed helpers ----------

    def classic(self, alpha: ArrayLike, omega: ArrayLike, r: ArrayLike) -> np.ndarray:
        """Return α×r + ω×(ω×r) (tangential + centripetal)."""
        return ClassicAccel.at_point(np.asarray(alpha, float),
                                     np.asarray(omega, float),
                                     np.asarray(r, float))

    def euler_alpha_zyx(
        self,
        angles: ArrayLike,
        rates: ArrayLike,
        accels: ArrayLike,
    ) -> np.ndarray:
        """ZYX Euler angular_velocity acceleration_kinematics (9.127–9.131 analogue)."""
        return EulerKinematics("ZYX").alpha(np.asarray(angles, float),
                                            np.asarray(rates, float),
                                            np.asarray(accels, float))

    def quaternion_SB(self, q: ArrayLike, qd: ArrayLike, qdd: ArrayLike) -> np.ndarray:
        """Quaternion acceleration_kinematics transform S_B (9.175–9.181)."""
        return QuaternionKinematics().S_B(np.asarray(q, float),
                                          np.asarray(qd, float),
                                          np.asarray(qdd, float))

    def mixed_G_of_B(
        self,
        R: np.ndarray,
        omega: ArrayLike,
        alpha: ArrayLike,
        r: ArrayLike,
        vB: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Representative mixed acceleration_kinematics helper (subset of 9.400–9.426).

        Returns
        -------
        (a_BG, a_GB) : B-expression of G-accel, and G-expression of B-accel.
        """
        return MixedAcceleration.G_of_B(np.asarray(R, float),
                                        np.asarray(omega, float),
                                        np.asarray(alpha, float),
                                        np.asarray(r, float),
                                        np.asarray(vB, float))

    # ======================================================================
    # Problem API (used by CLI/tests)
    # ======================================================================

    # --- model parsing ---

    def _build_chain_from_model(self, model: Mapping[str, Any]) -> ChainKinematics:
        """
        Supports:
          - {"kind": "planar2r", "l1": ..., "l2": ...}
        (Extend here when you add more backends/designs.)
        """
        kind = str(model.get("kind", "")).lower()
        if kind == "planar2r":
            l1 = float(model["l1"]); l2 = float(model["l2"])
            return self.preset_planar_2r(l1, l2)
        raise ValueError(f"Unsupported model.kind for acceleration_kinematics: {kind}")

    # --- operation routing ---

    def _op_forward(self, model: Mapping[str, Any], payload: Mapping[str, Any]) -> List[float]:
        chain = self._build_chain_from_model(model)
        q = payload.get("q", [0.0, 0.0])
        qd = payload.get("qd", [0.0, 0.0])
        qdd = payload.get("qdd", [0.0, 0.0])
        res, _ = self.forward_accel(chain, q, qd, qdd)
        return res.xdd.tolist()

    def _op_inverse(self, model: Mapping[str, Any], payload: Mapping[str, Any]) -> List[float]:
        chain = self._build_chain_from_model(model)
        q = payload.get("q", [0.0, 0.0])
        qd = payload.get("qd", [0.0, 0.0])
        xdd = payload.get("xdd", [0.0, 0.0])
        damping = float(payload.get("damping", 1e-8))
        res, _ = self.inverse_accel(chain, q, qd, xdd, damping=damping)
        return res.qdd.tolist()

    def _op_classic(self, payload: Mapping[str, Any]) -> List[float]:
        a = payload.get("alpha", [0.0, 0.0, 0.0])
        w = payload.get("omega", [0.0, 0.0, 0.0])
        r = payload.get("r", [0.0, 0.0, 0.0])
        return self.classic(a, w, r).tolist()

    def _op_euler_alpha(self, payload: Mapping[str, Any]) -> List[float]:
        return self.euler_alpha_zyx(payload.get("angles", [0,0,0]),
                                    payload.get("rates", [0,0,0]),
                                    payload.get("accels", [0,0,0])).tolist()

    def _op_quat_sb(self, payload: Mapping[str, Any]) -> List[float]:
        return self.quaternion_SB(payload.get("q", [1,0,0,0]),
                                  payload.get("qd", [0,0,0,0]),
                                  payload.get("qdd", [0,0,0,0])).tolist()

    def _op_mixed(self, payload: Mapping[str, Any]) -> Mapping[str, List[float]]:
        R = np.asarray(payload.get("R", np.eye(3)), float)
        w = payload.get("omega", [0,0,0])
        a = payload.get("alpha", [0,0,0])
        r = payload.get("r", [0,0,0])
        vB= payload.get("vB",[0,0,0])
        aBG, aGB = self.mixed_G_of_B(R, w, a, r, vB)
        return {"a_BG": aBG.tolist(), "a_GB": aGB.tolist()}

    # --- public problem solvers ---

    def solve(self, problem: Mapping[str, Any]) -> Any:
        """
        Solve a single problem dict of the form:

          {"op": "forward_kinematics|inverse_kinematics|classic|euler_alpha|quat_SB|mixed",
           "model": {...},   # only for forward_kinematics|inverse_kinematics
           "payload": {...}} # op-specific arguments

        Returns a JSON-serializable result (lists).
        """
        op = str(problem.get("op", "")).lower()
        model = problem.get("model", {})
        payload = problem.get("payload", {})

        if op == "forward_kinematics":
            return self._op_forward(model, payload)
        if op == "inverse_kinematics":
            return self._op_inverse(model, payload)
        if op == "classic":
            return self._op_classic(payload)
        if op == "euler_alpha":
            return self._op_euler_alpha(payload)
        if op == "quat_sb":
            return self._op_quat_sb(payload)
        if op == "mixed":
            return self._op_mixed(payload)

        raise ValueError(f"Unsupported op: {op}")

    def solve_from_path(self, path: PathLike) -> Any:
        """Load a single problem JSON file and solve it."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.solve(payload)

    def solve_batch(self, problems: Sequence[Mapping[str, Any]]) -> List[Any]:
        """Solve a batch of heterogeneous problems (results in given order)."""
        return [self.solve(p) for p in problems]

    # ---------- Class Diagram helpers ----------

    def class_diagram_dot(self) -> str:
        """
        Return a Graphviz DOT class diagram covering the acceleration_kinematics package.
        Uses optional `acceleration_kinematics.tools.diagram.render_dot` if available.
        """
        if _render_dot is None:
            return "// diagram helper not installed"
        return _render_dot(
            packages=("acceleration_kinematics.core", "acceleration_kinematics.io",
                      "acceleration_kinematics.utils", "acceleration_kinematics.app",
                      "acceleration_kinematics.backends.base",
                      "acceleration_kinematics.backends.numpy_backend")
        )

    # ---- Mermaid generator (introspects live classes) ----

    def generate_class_diagram(self, out_markdown: Path) -> Path:
        """
        Discover classes in the acceleration_kinematics package and emit a Markdown file with a
        Mermaid `classDiagram` block. We include classes, inheritance, and basic
        associations inferred from type annotations.
        """
        packages = ("acceleration_kinematics.core", "acceleration_kinematics.io", "acceleration_kinematics.utils",
                    "acceleration_kinematics.app", "acceleration_kinematics.backends.base",
                    "acceleration_kinematics.backends.numpy_backend")

        # Discover classes by package prefix
        classes: List[type] = []
        for modname in packages:
            try:
                mod = importlib.import_module(modname)
            except Exception:
                continue
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if getattr(obj, "__module__", "").startswith(packages):
                    classes.append(obj)

        qname = lambda c: f"{c.__module__}.{c.__name__}"
        in_scope = {qname(c): c for c in classes}

        def mermaid_id(c: type) -> str:
            return f'{c.__module__.replace(".", "_")}__{c.__name__}'

        lines: List[str] = []
        lines.append("# Acceleration module class diagram\n")
        lines.append("```mermaid")
        lines.append("classDiagram")

        # nodes
        for c in sorted(classes, key=qname):
            lines.append(f"class {mermaid_id(c)} as {c.__name__}")

        # inheritance
        names = set(in_scope.keys())
        for c in classes:
            for b in c.__bases__:
                if hasattr(b, "__name__"):
                    bqn = qname(b)
                    if bqn in names:
                        lines.append(f"{mermaid_id(c)} --|> {mermaid_id(in_scope[bqn])}")

        # basic associations via annotations (Optional/Sequence supported)
        def _is_optional_of(tp, target: type) -> bool:
            origin = get_origin(tp)
            if origin is None:
                return False
            if origin is Optional:
                return target in get_args(tp)
            if origin and str(origin).endswith("Union"):
                return target in get_args(tp)
            return False

        def _is_sequence_of(tp, target: type) -> bool:
            origin = get_origin(tp)
            if origin in (list, set, tuple):
                args = get_args(tp)
                return any(a is target for a in args)
            if origin and any(s in str(origin) for s in ("Sequence", "List", "Set")):
                args = get_args(tp)
                return len(args) == 1 and args[0] is target
            return False

        for c in classes:
            ann = getattr(c, "__annotations__", {}) or {}
            if not ann:
                continue
            for attr, tp in ann.items():
                targets = []
                if isinstance(tp, type) and qname(tp) in names:
                    targets = [tp]
                else:
                    for cand in classes:
                        if _is_sequence_of(tp, cand) or _is_optional_of(tp, cand):
                            targets = [cand]
                            break
                        origin = get_origin(tp)
                        args = get_args(tp)
                        if origin is None and args:
                            for a in args:
                                if isinstance(a, type) and qname(a) in names:
                                    targets = [a]
                                    break
                for t in targets:
                    lines.append(f"{mermaid_id(c)} --> {mermaid_id(t)} : {attr}")

        lines.append("```")
        content = "\n".join(lines)
        out_markdown.parent.mkdir(parents=True, exist_ok=True)
        out_markdown.write_text(content, encoding="utf-8")
        return out_markdown


__all__ = ["AccelApp", "AppInfo"]
