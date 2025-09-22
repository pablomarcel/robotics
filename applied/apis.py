# applied/apis.py
"""
Python APIs for the **applied dynamics** package.

Layers
------
1) AppliedDynamicsAPI — test-oriented façade exposing quick symbolic derivations.
   Returns `applied.utils.Result` payloads, keeping a single LagrangeEngine.

2) AppliedService — optional, app-oriented façade mirroring the inverse module
   style. Useful for batch runs, file I/O helpers, and diagram generation without
   tying to a web framework.

Design notes
------------
- Avoid circular imports by importing `AppliedApp` only inside methods that use it.
- Keep return shapes stable for tests (Result objects with fields like EOM/K/V/M).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import sympy as sp

from .dynamics import LagrangeEngine, mass_matrix_from_lagrangian
from .models import (
    SimplePendulum,
    SphericalPendulum,
    Planar2R,
    CartPendulumAbsorber,
)
from .core import FrameState
from .utils import Result, OUT_DIR


# -----------------------------------------------------------------------------
# Back-compat, test-facing façade
# -----------------------------------------------------------------------------

@dataclass
class AppliedDynamicsAPI:
    """
    Lightweight façade used by tests to run quick symbolic derivations.

    Methods return `Result(name, data)` where `data` typically includes:
      - "EOM": sympy Matrix (Euler–Lagrange equations, left-hand side)
      - "K":   kinetic energy (Expr)
      - "V":   potential energy (Expr)
      - "M":   mass matrix M(q) when relevant
    """
    engine: LagrangeEngine = field(default_factory=LagrangeEngine)

    def derive_simple_pendulum(self) -> Result:
        # IMPORTANT: use symbols without assumptions so tests comparing by name match
        m, l, g = sp.symbols("m l g")
        model = SimplePendulum(m, l, g)
        q, qd, t = model.lagrangian_state()
        fs = FrameState(q, qd)
        K = model.kinetic(fs)
        V = model.potential(fs)
        eoms = self.engine.equations_of_motion([model.th], t, K, V)
        M = mass_matrix_from_lagrangian(K, list(qd))
        return Result("simple_pendulum", {"EOM": eoms, "K": K, "V": V, "M": M})

    def derive_spherical_pendulum(self) -> Result:
        m, l, g = sp.symbols("m l g", positive=True)
        model = SphericalPendulum(m, l, g)
        q, qd, t = model.lagrangian_state()
        fs = FrameState(q, qd)
        K = model.kinetic(fs)
        V = model.potential(fs)
        eoms = self.engine.equations_of_motion([model.th, model.ph], t, K, V)
        return Result("spherical_pendulum", {"EOM": eoms, "K": K, "V": V})

    def derive_planar_2r(self) -> Result:
        m1, m2, l1, l2, g = sp.symbols("m1 m2 l1 l2 g", positive=True)
        model = Planar2R(m1, m2, l1, l2, g)
        q, qd, t = model.lagrangian_state()
        fs = FrameState(q, qd)
        K = model.kinetic(fs)
        V = model.potential(fs)
        eoms = self.engine.equations_of_motion([model.th1, model.th2], t, K, V)
        M = mass_matrix_from_lagrangian(K, list(qd))
        return Result("planar_2r", {"EOM": eoms, "K": K, "V": V, "M": M})

    def derive_cart_absorber(self) -> Result:
        M, m, l, k, g = sp.symbols("M m l k g", positive=True)
        model = CartPendulumAbsorber(M, m, l, k, g)
        q, qd, t = model.lagrangian_state()
        fs = FrameState(q, qd)
        K = model.kinetic(fs)
        V = model.potential(fs)
        eoms = self.engine.equations_of_motion([model.x, model.th], t, K, V)
        return Result("cart_absorber", {"EOM": eoms, "K": K, "V": V})


# -----------------------------------------------------------------------------
# App-oriented façade (optional; mirrors inverse module style)
# -----------------------------------------------------------------------------

@dataclass
class AppliedService:
    """
    Higher-level façade around `AppliedApp`. Not required by current tests,
    but useful for parity with the inverse module’s service and for scripting.
    """

    def run_all(self) -> List[Result]:
        """Run the app’s bundled derivations and return their results in-memory."""
        from .app import AppliedApp
        app = AppliedApp()
        return app.run_all()

    def save_result_json(self, name: str, payload: Dict[str, Any], out_dir: Optional[Path] = None) -> str:
        """
        Save a dict payload as JSON under applied/out (or a custom out_dir).
        Returns the full path as a string.
        """
        out = (out_dir or OUT_DIR) / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
        return str(out)


__all__ = ["AppliedDynamicsAPI", "AppliedService"]
