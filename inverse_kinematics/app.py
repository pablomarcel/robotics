# inverse_kinematics/app.py
"""
High-level façade for the inverse_kinematics-kinematics package.

This class mirrors the forward_kinematics module’s app: it centralizes file I/O,
preset builders, and IK solvers (closed-form and iterative) behind a
small, testable API. Importantly, it **does not** import `inverse_kinematics.apis`
to avoid circular imports with any service layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, get_args, get_origin

import importlib
import inspect
import json
import numpy as np

from .utils import timed
from .core import (
    Transform,
    SerialChain,
    IterativeIK,
    ClosedFormIK,
)
from . import io as io_mod
from . import design as design_mod
from .tools.diagram import render_dot as _render_dot

Number = Union[int, float]
ArrayLike = Union[Sequence[Number], np.ndarray]
PathLike = Union[str, Path]


@dataclass(frozen=True)
class AppInfo:
    """Immutable metadata about the application."""
    name: str = "inverse_kinematics-kinematics"
    version: str = "0.1.0"
    homepage: str = "https://example.local/inverse"  # placeholder


class InverseApp:
    """
    High-level application façade for inverse_kinematics kinematics workflows.

    Responsibilities
    ----------------
    * Load/validate robot specs and build :class:`SerialChain`.
    * Closed-form IK helpers for supported models (e.g. planar 2R).
    * Iterative IK (Newton–Raphson / DLS) for general chains.
    * Preset builders mirroring those in :mod:`inverse_kinematics.design`.
    * Class diagram export (DOT text) and Mermaid markdown emitter.
    * Problem-API utilities used by the CLI/tests (solve / batch / from-file).
    """

    def __init__(self, in_dir: Optional[PathLike] = None, out_dir: Optional[PathLike] = None):
        self.info = AppInfo()
        self.in_dir = Path(in_dir) if in_dir else Path("inverse_kinematics/in")
        self.out_dir = Path(out_dir) if out_dir else Path("inverse_kinematics/out")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Loading & Validation ----------

    def load_robot(self, path: PathLike, *, validate: bool = True) -> SerialChain:
        """
        Load a robot specification (JSON or YAML) and build a :class:`SerialChain`.
        """
        p = Path(path)
        spec = io_mod.load_spec(p)  # JSON/YAML auto-detected
        if validate:
            io_mod.validate_spec(spec, io_mod.robot_schema())
        return io_mod.build_chain_from_spec(spec)

    def validate_file(self, path: PathLike) -> Tuple[bool, Optional[str]]:
        """
        Validate a robot specification file against the JSON Schema.
        """
        try:
            spec = io_mod.load_spec(path)
            io_mod.validate_spec(spec, io_mod.robot_schema())
            return True, None
        except Exception as exc:
            return False, str(exc)

    # ---------- Presets (Design helpers) ----------

    def preset_planar_2r(self, l1: Number, l2: Number) -> SerialChain:
        return design_mod.planar_2r(l1, l2)

    def preset_scara(self, l1: Number, l2: Number, d: Number = 0.0, wrist_rotary: bool = False) -> SerialChain:
        # Optional: preserve API but avoid import errors if not implemented
        if hasattr(design_mod, "scara"):
            return design_mod.scara(l1, l2, d_home=d, wrist_rotary=wrist_rotary)  # type: ignore[no-any-return]
        raise NotImplementedError("design.scara is not available in this build.")

    def preset_spherical_wrist(self, wrist_type: int, d_tool: Number = 0.0) -> SerialChain:
        return design_mod.spherical_wrist(wrist_type=wrist_type, d_tool=d_tool)

    # ---------- IK: closed-form convenience wrappers ----------

    def solve_closed_form_planar2r(self, x: float, y: float, l1: float, l2: float) -> List[np.ndarray]:
        """Closed-form planar 2R IK; returns two branches when reachable."""
        return ClosedFormIK.planar_2r(x, y, l1, l2)

    def solve_closed_form_wrist(self, R: np.ndarray) -> List[np.ndarray]:
        """Closed-form spherical wrist orientation_kinematics."""
        return ClosedFormIK.spherical_wrist(R)

    # ---------- IK: iterative (general chains) ----------

    @timed
    def solve_iterative(
        self,
        chain: SerialChain,
        T_target: Union[np.ndarray, Transform],
        q0: ArrayLike,
        *,
        tol: float = 1e-6,
        max_iters: int = 50,
        damping: Optional[float] = None,
        position_weight: float = 1.0,       # reserved for future use
        orientation_weight: float = 1.0,    # reserved for future use
        space: str = "space",
    ):
        """
        Newton–Raphson / Damped-Least-Squares IK.

        Parameters
        ----------
        chain : SerialChain
        T_target : 4×4 matrix or Transform
        q0 : initial joint vector
        tol : stopping tolerance on twist norm
        max_iters : iteration cap
        damping : if provided, LM damping parameter (λ)
        space : "space" or "body" Jacobian convention for the step

        Returns
        -------
        An object with attribute `.q` (np.ndarray), containing the first/best
        joint solution returned by the underlying IterativeIK solver. This
        preserves backward-compat behavior expected elsewhere in the app.
        """
        T = T_target.as_matrix() if isinstance(T_target, Transform) else np.asarray(T_target, float)

        # Instantiate per the IterativeIK API used in tests
        kwargs = dict(tol=float(tol), itmax=int(max_iters), space=str(space))
        if damping is not None:
            kwargs["lambda_damp"] = float(damping)

        solver = IterativeIK(**kwargs)
        sols = solver.solve(chain, T, np.asarray(q0, float).reshape(-1))
        if not sols:
            # Normalize to a consistent shape: expose a `.q` that is empty
            class _Res:
                def __init__(self): self.q = np.asarray([], float)
            return _Res()

        q_best = np.asarray(sols[0], float).reshape(-1)

        class _Res:
            def __init__(self, q): self.q = q
        return _Res(q_best)

    # ======================================================================
    # Problem API (used by CLI/tests)
    # ======================================================================

    def _build_chain_from_problem_model(self, model: Mapping[str, Any]) -> SerialChain:
        """
        Supports:
          - {"kind": "planar2r", "l1": ..., "l2": ...}
          - {"kind": "wrist", "wrist_type": 1|2|3, "d_tool": float=0.0}
          - {"kind": "spec", "spec": <dict or path>}
        """
        kind = str(model.get("kind", "")).lower()
        if kind == "planar2r":
            l1 = float(model["l1"])
            l2 = float(model["l2"])
            return self.preset_planar_2r(l1, l2)
        if kind == "wrist":
            wt = int(model.get("wrist_type", 1))
            d_tool = float(model.get("d_tool", 0.0))
            return self.preset_spherical_wrist(wt, d_tool)
        if kind == "spec":
            spec = model.get("spec")
            if isinstance(spec, (str, Path)):
                return self.load_robot(spec, validate=True)
            if isinstance(spec, Mapping):
                io_mod.validate_spec(spec, io_mod.robot_schema())
                return io_mod.build_chain_from_spec(spec)
            raise ValueError("model.kind='spec' requires 'spec' as dict or path string.")
        raise ValueError(f"Unsupported model.kind: {kind}")

    def _parse_pose_to_T(self, pose: Mapping[str, Any]) -> np.ndarray:
        """
        Accepts either:
          - {"x": X, "y": Y} (z=0)
          - {"T": [[...],[...],[...],[...]]}
        """
        if "T" in pose:
            T = np.asarray(pose["T"], float)
            if T.shape != (4, 4):
                raise ValueError("pose.T must be 4x4")
            return T
        if "x" in pose and "y" in pose:
            T = np.eye(4, dtype=float)
            T[0, 3] = float(pose["x"])
            T[1, 3] = float(pose["y"])
            return T
        raise ValueError("pose must include either 'T' (4x4) or 'x' and 'y'")

    def _extract_l1_l2_from_chain(self, chain: SerialChain) -> Tuple[float, float]:
        """
        Robustly extract (l1, l2) from a standard-DH planar-2R chain where
        link lengths live in a1 and a2.
        """
        links = chain.links()
        if len(links) < 2:
            raise ValueError("Planar2R chain must have at least two links.")
        l1 = float(getattr(links[0], "a", 0.0))
        l2 = float(getattr(links[1], "a", 0.0))
        return l1, l2

    def _solve_planar2r_gateway(
        self,
        chain: SerialChain,
        pose: Mapping[str, Any],
        method: Mapping[str, Any],
    ) -> List[List[float]]:
        """
        Helper used by `solve_planar2r` and generic `solve(problem)`.
        """
        meth = str(method.get("method", "analytic")).lower()
        if meth not in {"analytic", "iterative"}:
            raise ValueError("method.method must be 'analytic' or 'iterative'")

        if meth == "analytic":
            # Use closed-form helper for planar 2R
            if "T" in pose:
                x, y = float(pose["T"][0][3]), float(pose["T"][1][3])
            else:
                x, y = float(pose["x"]), float(pose["y"])
            l1, l2 = self._extract_l1_l2_from_chain(chain)
            sols = ClosedFormIK.planar_2r(x, y, l1, l2)
            return [s.tolist() for s in sols]

        # ---------------------- iterative branch ----------------------
        tol = float(method.get("tol", 1e-6))
        itmax = int(method.get("itmax", method.get("max_iters", 200)))
        lam = method.get("lambda", method.get("lambda_damp", None))
        lamf = None if lam is None else float(lam)
        space = str(method.get("space", "space")).lower()

        Tdes = self._parse_pose_to_T(pose)

        # Seed handling: honor user-provided q0; otherwise try a closed-form seed.
        if "q0" in method and method["q0"] is not None:
            q0 = np.asarray(method["q0"], float).reshape(-1)
        else:
            # Convert pose to (x, y) for planar seeding.
            if "T" in pose:
                x_seed, y_seed = float(pose["T"][0][3]), float(pose["T"][1][3])
            else:
                x_seed, y_seed = float(pose["x"]), float(pose["y"])
            try:
                l1, l2 = self._extract_l1_l2_from_chain(chain)
                cf_solutions = ClosedFormIK.planar_2r(x_seed, y_seed, l1, l2)
                if cf_solutions:
                    q0 = np.asarray(cf_solutions[0], float).reshape(-1)
                else:
                    q0 = np.zeros(chain.n(), float)
            except Exception:
                q0 = np.zeros(chain.n(), float)

        res = self.solve_iterative(chain, Tdes, q0, tol=tol, max_iters=itmax, damping=lamf, space=space)
        return [res.q.tolist()]

    # ---------------- Public problem methods ----------------

    def solve_planar2r(
        self,
        *,
        l1: float,
        l2: float,
        x: float,
        y: float,
        method: str = "analytic",
        q0: Optional[Sequence[float]] = None,
        tol: float = 1e-6,
        itmax: int = 200,
        lambda_damp: float = 1e-3,
        space: str = "space",
    ) -> List[List[float]]:
        """
        Solve planar-2R IK for a single (x, y) target using either the analytic
        or the iterative solver. Returns a **list of solutions**, each a list
        [q1, q2] in radians.
        """
        chain = design_mod.planar_2r(float(l1), float(l2))
        pose = {"x": float(x), "y": float(y)}
        meth = {"method": str(method).lower()}
        if meth["method"] == "iterative":
            meth.update({
                "tol": float(tol),
                "itmax": int(itmax),
                "lambda": float(lambda_damp),
                "q0": (q0 or [0.0, 0.0]),
                "space": space
            })
        return self._solve_planar2r_gateway(chain, pose, meth)

    def solve(self, problem: Mapping[str, Any]) -> List[List[float]]:
        """
        Solve a single problem dict of the form:
            {"model": {...}, "method": {...}, "pose": {...}}
        """
        model = problem.get("model", {})
        pose = problem.get("pose", {})
        method = problem.get("method", {"method": "analytic"})
        chain = self._build_chain_from_problem_model(model)

        if str(model.get("kind", "")).lower() == "planar2r":
            return self._solve_planar2r_gateway(chain, pose, method)

        if str(method.get("method", "iterative")).lower() == "iterative":
            Tdes = self._parse_pose_to_T(pose)
            tol = float(method.get("tol", 1e-6))
            itmax = int(method.get("itmax", method.get("max_iters", 200)))
            lam = method.get("lambda", method.get("lambda_damp", None))
            lamf = None if lam is None else float(lam)
            q0 = np.asarray(method.get("q0", [0.0] * chain.n()), float).reshape(-1)
            space = str(method.get("space", "space")).lower()
            res = self.solve_iterative(chain, Tdes, q0, tol=tol, max_iters=itmax, damping=lamf, space=space)
            return [res.q.tolist()]

        raise ValueError("Unsupported combination; try method='iterative' for generic chains.")

    def solve_from_path(self, path: PathLike) -> List[List[float]]:
        """Load a single problem JSON file and solve it."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.solve(payload)

    def solve_batch(
        self,
        *,
        model: Mapping[str, Any],
        poses: Sequence[Mapping[str, Any]],
        method: Mapping[str, Any],
    ) -> List[List[List[float]]]:
        """
        Solve a batch of poses for a fixed model & method.
        Returns list-of-lists: one list per pose, each containing solutions.
        """
        chain = self._build_chain_from_problem_model(model)
        out: List[List[List[float]]] = []
        for pose in poses:
            if str(model.get("kind", "")).lower() == "planar2r":
                sols = self._solve_planar2r_gateway(chain, pose, method)
            else:
                Tdes = self._parse_pose_to_T(pose)
                tol = float(method.get("tol", 1e-6))
                itmax = int(method.get("itmax", method.get("max_iters", 200)))
                lam = method.get("lambda", method.get("lambda_damp", None))
                lamf = None if lam is None else float(lam)
                q0 = np.asarray(method.get("q0", [0.0] * chain.n()), float).reshape(-1)
                space = str(method.get("space", "space")).lower()
                res = self.solve_iterative(chain, Tdes, q0, tol=tol, max_iters=itmax, damping=lamf, space=space)
                sols = [res.q.tolist()]
            out.append(sols)
        return out

    # ---------- Class Diagram helpers ----------

    def class_diagram_dot(self) -> str:
        """Return a Graphviz DOT class diagram covering the inverse_kinematics package."""
        return _render_dot(
            packages=("inverse_kinematics.core", "inverse_kinematics.io", "inverse_kinematics.design", "inverse_kinematics.utils", "inverse_kinematics.app")
        )

    # ---- Mermaid generator (introspects live classes) ----

    def generate_class_diagram(self, out_markdown: Path) -> Path:
        """
        Discover classes in the inverse_kinematics package and emit a Markdown file with a
        Mermaid `classDiagram` block. We include classes, inheritance, and basic
        associations inferred from type annotations.

        The CLI tests only assert that a valid Mermaid block exists, but this
        implementation reflects the actual code instead of a static stub.
        """
        packages = ("inverse_kinematics.core", "inverse_kinematics.io", "inverse_kinematics.design", "inverse_kinematics.utils", "inverse_kinematics.app")

        # Discover classes by package prefix
        classes: List[type] = []
        for modname in packages:
            mod = importlib.import_module(modname)
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if getattr(obj, "__module__", "").startswith(packages):
                    classes.append(obj)

        # Index helpers
        qname = lambda c: f"{c.__module__}.{c.__name__}"
        in_scope = {qname(c): c for c in classes}

        def mermaid_id(c: type) -> str:
            # Mermaid identifiers are simple; avoid dots
            return f'{c.__module__.replace(".", "_")}__{c.__name__}'

        # Build lines
        lines: List[str] = []
        lines.append("# Inverse module class diagram\n")
        lines.append("```mermaid")
        lines.append("classDiagram")

        # Declare classes
        for c in sorted(classes, key=qname):
            lines.append(f"class {mermaid_id(c)} as {c.__name__}")

        # Inheritance edges
        in_scope_names = set(in_scope.keys())
        for c in classes:
            cid = mermaid_id(c)
            for b in c.__bases__:
                if hasattr(b, "__name__"):
                    bqn = qname(b)
                    if bqn in in_scope_names:
                        lines.append(f"{cid} --|> {mermaid_id(in_scope[bqn])}")

        # Simple association edges from annotations (Optional/Sequence supported)
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
                # direct
                targets = []
                if isinstance(tp, type) and qname(tp) in in_scope_names:
                    targets = [tp]
                else:
                    # generics
                    for cand in classes:
                        if _is_sequence_of(tp, cand) or _is_optional_of(tp, cand):
                            targets = [cand]
                            break
                        # raw Union/args scan
                        origin = get_origin(tp)
                        args = get_args(tp)
                        if origin is None and args:
                            for a in args:
                                if isinstance(a, type) and qname(a) in in_scope_names:
                                    targets = [a]
                                    break
                for t in targets:
                    lines.append(f"{mermaid_id(c)} --> {mermaid_id(t)} : {attr}")

        lines.append("```")
        content = "\n".join(lines)

        out_markdown.parent.mkdir(parents=True, exist_ok=True)
        out_markdown.write_text(content, encoding="utf-8")
        return out_markdown


__all__ = ["InverseApp", "AppInfo"]
