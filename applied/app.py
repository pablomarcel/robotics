# applied/app.py
"""
High-level façade for the applied-dynamics package.

This module **does not** import `applied.apis` at import time (to avoid cycles).
It offers:
- System presets via :mod:`applied.design`
- Symbolic derivation (Lagrange engine) for any :class:`applied.core.System`
- Convenience wrappers for common systems (pendulums, planar 2R, cart+absorber)
- Optional numeric simulation hooks (adapter over :mod:`applied.integrators`)
- Problem-API helpers used by CLI/tests
- Class diagram utilities via :mod:`applied.tools.diagram`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, get_args, get_origin
import importlib
import inspect
import json

import numpy as np
import sympy as sp

from .utils import timed
from . import io as io_mod
from . import design as design_mod
from .core import System, FrameState
from .tools.diagram import render_dot as _render_dot

Number = Union[int, float]
ArrayLike = Union[Sequence[Number], np.ndarray]
PathLike = Union[str, Path]

# Re-export the global time symbol used by models/engines (helps tests)
t = sp.symbols("t", real=True)


# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppInfo:
    """Immutable metadata about the application."""
    name: str = "applied-dynamics"
    version: str = "0.1.0"
    homepage: str = "https://example.local/applied"  # placeholder


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DerivationResult:
    """Symbolic derivation artifacts for a system."""
    name: str
    q: sp.Matrix
    qd: sp.Matrix
    L: sp.Expr          # Lagrangian K - V
    K: sp.Expr          # kinetic
    V: sp.Expr          # potential
    Q: sp.Matrix        # non-potential generalized forces
    EOM: sp.Matrix      # equations of motion_kinematics (tau form: d/dt(∂L/∂q̇)-∂L/∂q = Q)


# ---------------------------------------------------------------------------
# Application façade
# ---------------------------------------------------------------------------

class AppliedApp:
    """
    High-level façade for *applied dynamics* workflows.

    Responsibilities
    ----------------
    * Build :class:`System` presets from :mod:`applied.design`.
    * Derive symbolic equations of motion_kinematics using the Lagrange engine.
    * Optional numerical simulation front-end (wrapping `applied.integrators`).
    * Problem-API helpers used by CLI/tests (derive / simulate / batch).
    * Class diagram helpers (DOT / Mermaid Markdown).
    """

    def __init__(self, in_dir: Optional[PathLike] = None, out_dir: Optional[PathLike] = None):
        self.info = AppInfo()
        self.in_dir = Path(in_dir) if in_dir else Path("applied/in")
        self.out_dir = Path(out_dir) if out_dir else Path("applied/out")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Presets (Design helpers)
    # -----------------------------------------------------------------------

    def preset(self, name: str, **params: Any) -> System:
        """
        Generic preset builder routed to :class:`applied.design.DesignLibrary`.
        """
        lib = design_mod.DesignLibrary()
        return lib.create(name, **params)

    # Named convenience creators (thin wrappers for readability/CLI ergonomics)

    def preset_simple_pendulum(self, m: Number, l: Number, g: Number) -> System:
        return self.preset("simple_pendulum", m=m, l=l, g=g)

    def preset_spherical_pendulum(self, m: Number, l: Number, g: Number) -> System:
        return self.preset("spherical_pendulum", m=m, l=l, g=g)

    def preset_planar_2r(
        self, m1: Number, m2: Number, l1: Number, l2: Number, g: Number
    ) -> System:
        return self.preset("planar_2r", m1=m1, m2=m2, l1=l1, l2=l2, g=g)

    def preset_cart_absorber(
        self, M: Number, m: Number, l: Number, k: Number, g: Number
    ) -> System:
        return self.preset("cart_pendulum_absorber", M=M, m=m, l=l, k=k, g=g)

    # -----------------------------------------------------------------------
    # Derivation
    # -----------------------------------------------------------------------

    def _lagrange_engine(self):
        # Local import avoids cycles and keeps optional deps lazy.
        from .dynamics import LagrangeEngine
        return LagrangeEngine()

    @timed
    def derive(self, sys: System) -> DerivationResult:
        """
        Symbolically derive the EOM for any :class:`System` using the
        :class:`applied.dynamics.LagrangeEngine`.
        """
        q, qd, _ = sys.lagrangian_state()
        fs = FrameState(q, qd)

        K = sp.simplify(sys.kinetic(fs))
        V = sp.simplify(sys.potential(fs))
        L = sp.simplify(K - V)
        Q = sp.Matrix(sys.generalized_forces(fs))

        engine = self._lagrange_engine()
        # recover function constructors from q (theta(t) -> theta)
        q_funcs: List[sp.Function] = []
        for expr in q:
            if isinstance(expr, sp.AppliedUndef):
                q_funcs.append(expr.func)  # type: ignore[arg-type]
            else:
                # fallback generic name
                q_funcs.append(sp.Function("q"))

        EOM = sp.Matrix(engine.equations_of_motion(q_funcs, t, K, V, Q))

        return DerivationResult(
            name=getattr(sys, "name", sys.__class__.__name__),
            q=q, qd=qd, L=L, K=K, V=V, Q=Q, EOM=EOM
        )

    # Named convenience derivations (used by CLI/tests)

    def derive_simple_pendulum(self, m: Number, l: Number, g: Number) -> DerivationResult:
        return self.derive(self.preset_simple_pendulum(m, l, g))

    def derive_spherical_pendulum(self, m: Number, l: Number, g: Number) -> DerivationResult:
        return self.derive(self.preset_spherical_pendulum(m, l, g))

    def derive_planar_2r(
        self, m1: Number, m2: Number, l1: Number, l2: Number, g: Number
    ) -> DerivationResult:
        return self.derive(self.preset_planar_2r(m1, m2, l1, l2, g))

    def derive_cart_absorber(
        self, M: Number, m: Number, l: Number, k: Number, g: Number
    ) -> DerivationResult:
        return self.derive(self.preset_cart_absorber(M, m, l, k, g))

    # -----------------------------------------------------------------------
    # Optional numeric simulation (adapter over applied.integrators)
    # -----------------------------------------------------------------------

    def _integrators(self):
        return importlib.import_module("applied.integrators")

    def simulate_symbolic(
        self,
        sys: System,
        t_span: Tuple[float, float],
        y0: Sequence[float],
        *,
        steps: int = 1000,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        args: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        Simulate a :class:`System` from its symbolic EOM by auto-lambdifying
        the ODE right-hand side (see :mod:`applied.integrators`).
        """
        integ = self._integrators()
        rhs = integ.RHS.from_system(sys)        # auto-builds a lambdified f(t, y)
        solver = integ.Solver(rhs, method=method, rtol=rtol, atol=atol)
        sol = solver.solve(t_span, y0, steps=steps, args=dict(args or {}))
        return dict(t=sol.t, y=sol.y, events=sol.events)

    # -----------------------------------------------------------------------
    # Problem-API (used by CLI/tests)
    # -----------------------------------------------------------------------

    def _build_system_from_problem(self, model: Mapping[str, Any]) -> System:
        """
        Accepts:
          {"kind": "simple_pendulum", "m":..,"l":..,"g":..}
          {"kind": "spherical_pendulum", ...}
          {"kind": "planar_2r", "m1":..,"m2":..,"l1":..,"l2":..,"g":..}
          {"kind": "cart_pendulum_absorber", "M":..,"m":..,"l":..,"k":..,"g":..}
          {"kind": "preset", "name": "<DesignLibrary key>", "params": {...}}
        """
        kind = str(model.get("kind", "")).lower()
        if kind == "preset":
            return self.preset(str(model["name"]), **dict(model.get("params", {})))
        if kind == "simple_pendulum":
            return self.preset_simple_pendulum(model["m"], model["l"], model["g"])
        if kind == "spherical_pendulum":
            return self.preset_spherical_pendulum(model["m"], model["l"], model["g"])
        if kind == "planar_2r":
            return self.preset_planar_2r(model["m1"], model["m2"], model["l1"], model["l2"], model["g"])
        if kind == "cart_pendulum_absorber":
            return self.preset_cart_absorber(model["M"], model["m"], model["l"], model["k"], model["g"])
        raise ValueError(f"Unsupported model.kind: {kind}")

    def derive_problem(self, problem: Mapping[str, Any]) -> DerivationResult:
        """
        Derive EOM for a problem dict with a 'model' section.
        """
        sys = self._build_system_from_problem(problem.get("model", {}))
        return self.derive(sys)

    def derive_from_path(self, path: PathLike) -> DerivationResult:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.derive_problem(payload)

    # -----------------------------------------------------------------------
    # Batch helpers (e.g., run-all scripting)
    # -----------------------------------------------------------------------

    def run_all_to_out(self) -> List[DerivationResult]:
        """
        Derive a handful of canonical systems and dump artifacts to `applied/out`.
        """
        items = [
            ("simple_pendulum", dict(m=sp.Symbol("m"), l=sp.Symbol("l"), g=sp.Symbol("g"))),
            ("spherical_pendulum", dict(m=sp.Symbol("m"), l=sp.Symbol("l"), g=sp.Symbol("g"))),
            ("planar_2r", dict(m1=sp.Symbol("m1"), m2=sp.Symbol("m2"),
                               l1=sp.Symbol("l1"), l2=sp.Symbol("l2"), g=sp.Symbol("g"))),
            ("cart_pendulum_absorber", dict(M=sp.Symbol("M"), m=sp.Symbol("m"),
                                            l=sp.Symbol("l"), k=sp.Symbol("k"), g=sp.Symbol("g"))),
        ]
        results: List[DerivationResult] = []
        for name, params in items:
            sys = self.preset(name, **params)
            res = self.derive(sys)
            # Persist compact JSON (stringified SymPy)
            payload = {
                "name": res.name,
                "q": [str(x) for x in res.q],
                "qd": [str(x) for x in res.qd],
                "L": str(res.L),
                "K": str(res.K),
                "V": str(res.V),
                "Q": [str(x) for x in res.Q],
                "EOM": [str(x) for x in res.EOM],
            }
            out = self.out_dir / f"{name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            results.append(res)
        return results

    # -----------------------------------------------------------------------
    # Class Diagram helpers
    # -----------------------------------------------------------------------

    def class_diagram_dot(self) -> str:
        """Return Graphviz DOT for the applied package classes."""
        return _render_dot(
            packages=("applied.core", "applied.io", "applied.design", "applied.utils", "applied.app")
        )

    def generate_class_diagram(self, out_markdown: Path) -> Path:
        """
        Discover classes in the `applied` package and emit Mermaid `classDiagram`.
        """
        packages = ("applied.core", "applied.io", "applied.design", "applied.utils", "applied.app")

        # Discover classes by package prefix
        classes: List[type] = []
        for modname in packages:
            mod = importlib.import_module(modname)
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if getattr(obj, "__module__", "").startswith(packages):
                    classes.append(obj)

        qname = lambda c: f"{c.__module__}.{c.__name__}"
        in_scope = {qname(c): c for c in classes}

        def mid(c: type) -> str:
            return f'{c.__module__.replace(".", "_")}__{c.__name__}'

        lines: List[str] = []
        lines.append("# Applied module class diagram\n")
        lines.append("```mermaid")
        lines.append("classDiagram")

        for c in sorted(classes, key=qname):
            lines.append(f"class {mid(c)} as {c.__name__}")

        # Inheritance
        in_scope_names = set(in_scope.keys())
        for c in classes:
            cid = mid(c)
            for b in c.__bases__:
                if hasattr(b, "__name__"):
                    bqn = qname(b)
                    if bqn in in_scope_names:
                        lines.append(f"{cid} --|> {mid(in_scope[bqn])}")

        # Light association inference from type annotations
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
                targets: List[type] = []
                if isinstance(tp, type) and qname(tp) in in_scope_names:
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
                                if isinstance(a, type) and qname(a) in in_scope_names:
                                    targets = [a]
                                    break
                for tgt in targets:
                    lines.append(f"{mid(c)} --> {mid(tgt)} : {attr}")

        lines.append("```")
        out_markdown.parent.mkdir(parents=True, exist_ok=True)
        out_markdown.write_text("\n".join(lines), encoding="utf-8")
        return out_markdown


__all__ = ["AppliedApp", "AppInfo", "DerivationResult", "t"]
