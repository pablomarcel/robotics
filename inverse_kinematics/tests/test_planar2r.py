# inverse_kinematics/tests/test_planar2r.py
# Unit tests for the **analytic** 2R planar inverse_kinematics kinematics.

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from inverse_kinematics import design as design_mod
from inverse_kinematics.core import SerialChain, Transform

# Try both import spellings for the analytic solver in case the project
# exposes it under a class or a function. Mark one as None if missing.
try:
    # Preferred: class-based solver
    from inverse_kinematics.core import AnalyticPlanar2R  # type: ignore
except Exception:  # pragma: no cover - fallback path
    AnalyticPlanar2R = None  # type: ignore

try:
    # Fallback: functional API
    from inverse_kinematics.core import analytic_planar2r  # type: ignore
except Exception:  # pragma: no cover - fallback path
    analytic_planar2r = None  # type: ignore


def _ensure_solver():
    """
    Return a callable that implements the analytic planar 2R IK:
        solve(chain, x, y) -> List[np.ndarray]  (each ndarray shape (2,))
    We support either a class with .solve(...) or a free function.
    """
    if AnalyticPlanar2R is not None:
        solver = AnalyticPlanar2R()
        return lambda chain, x, y: solver.solve(chain, np.array([x, y, 0.0]))
    if analytic_planar2r is not None:
        return lambda chain, x, y: analytic_planar2r(chain, x, y)
    raise RuntimeError(
        "Neither inverse_kinematics.core.AnalyticPlanar2R nor inverse_kinematics.core.analytic_planar2r is available."
    )


def _fk_xy(chain: SerialChain, q: np.ndarray) -> Tuple[float, float]:
    T = chain.fkine(q).as_matrix()
    return float(T[0, 3]), float(T[1, 3])


def _near(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(a - b) <= tol


# ------------------------------ happy path ------------------------------

def test_two_solutions_for_reachable_target():
    """
    For l1=l2=1 and target (1,1), the classic 2R arm has **two** solutions
    (elbow-up and elbow-down). Verify both exist and match FK.
    """
    chain = design_mod.planar_2r(1.0, 1.0)
    solve = _ensure_solver()

    sols = solve(chain, 1.0, 1.0)
    assert isinstance(sols, list)
    assert len(sols) >= 2

    # FK check
    for q in sols:
        x, y = _fk_xy(chain, np.asarray(q, float))
        assert _near(x, 1.0, 1e-8) and _near(y, 1.0, 1e-8)

    # Distinguish elbow-up/down by q2 sign (convention: + for elbow-up typically)
    q2_vals = sorted([float(np.unwrap([0.0, q[1]])[1]) for q in sols])
    assert q2_vals[0] < 0.0 and q2_vals[-1] > 0.0


def test_edge_of_workspace_straight_configuration():
    """
    Target on the boundary (x=2, y=0) for l1=l2=1 should admit a straight-arm solution.
    """
    chain = design_mod.planar_2r(1.0, 1.0)
    solve = _ensure_solver()

    sols = solve(chain, 2.0, 0.0)
    assert len(sols) >= 1

    ok = False
    for q in sols:
        x, y = _fk_xy(chain, np.asarray(q, float))
        if _near(x, 2.0, 1e-8) and _near(y, 0.0, 1e-8):
            ok = True
            # Straight means q2 ≈ 0 (mod 2π). Accept wrap-around.
            q2 = float(q[1])
            q2_mod = math.atan2(math.sin(q2), math.cos(q2))
            assert abs(q2_mod) < 1e-6
    assert ok, "No solution produced the boundary target via FK."


def test_target_at_origin_folded_elbow():
    """
    With l1=l2=1 and target at the origin, a folded configuration exists (q2 ≈ π).
    """
    chain = design_mod.planar_2r(1.0, 1.0)
    solve = _ensure_solver()

    sols = solve(chain, 0.0, 0.0)
    assert len(sols) >= 1
    found_folded = False
    for q in sols:
        x, y = _fk_xy(chain, np.asarray(q, float))
        if _near(x, 0.0) and _near(y, 0.0):
            q2 = float(q[1])
            # Normalize to (-π, π]
            q2n = math.atan2(math.sin(q2), math.cos(q2))
            if abs(abs(q2n) - math.pi) < 1e-6:
                found_folded = True
    assert found_folded, "Expected a folded (q2≈π) configuration among solutions."


# ------------------------------ edge cases -------------------------------

def test_unreachable_target_returns_empty_or_raises():
    """
    For l1=l2=1, (x=3, y=0) is unreachable; either [] or a clear exception is acceptable.
    """
    chain = design_mod.planar_2r(1.0, 1.0)
    solve = _ensure_solver()

    try:
        sols = solve(chain, 3.0, 0.0)
        assert isinstance(sols, list)
        assert len(sols) == 0, "Unreachable target should yield no solutions."
    except ValueError:
        # Also acceptable API: raise ValueError on unreachable targets
        pass


def test_solutions_satisfy_law_of_cosines():
    """
    Validate the returned angles satisfy the 2R geometry for a known point.
    """
    l1, l2 = 1.0, 1.0
    chain = design_mod.planar_2r(l1, l2)
    solve = _ensure_solver()

    x, y = 0.3, 1.5
    r2 = x * x + y * y
    c2_expected = (r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)

    sols = solve(chain, x, y)
    assert len(sols) >= 1
    for q in sols:
        c2 = math.cos(float(q[1]))
        assert abs(c2 - c2_expected) < 1e-9

        # FK check too
        xf, yf = _fk_xy(chain, np.asarray(q, float))
        assert _near(xf, x, 1e-8) and _near(yf, y, 1e-8)
