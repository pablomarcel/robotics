# inverse/tests/test_apis_service.py
# Tests for the Python-facing API facade in inverse.apis (InverseService).

from __future__ import annotations

import numpy as np
import pytest

from inverse.apis import InverseService
from inverse import design as D


def _T_xy(x: float, y: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    return T


def test_service_validate_problem_ok():
    svc = InverseService()
    prob = {
        "model": {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
        "method": {"method": "analytic"},
        "pose": {"x": 1.0, "y": 1.0},
    }
    ok, err = svc.validate_problem(prob)
    assert ok and err is None


def test_service_validate_problem_fail():
    svc = InverseService()
    bad = {
        "model": {"kind": "planar2r", "l1": 1.0},  # missing l2
        "method": {"method": "analytic"},
        "pose": {"x": 1.0, "y": 1.0},
    }
    ok, err = svc.validate_problem(bad)
    assert not ok and isinstance(err, str)


def test_service_solve_problem_analytic_planar2r():
    svc = InverseService()
    prob = {
        "model": {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
        "method": {"method": "analytic"},
        "pose": {"x": 1.0, "y": 1.0},
    }
    sols = svc.solve_problem(prob)
    assert isinstance(sols, list) and len(sols) >= 2  # elbow-up & elbow-down

    # FK check with a chain built from the same model
    chain = D.planar_2r(1.0, 1.0)
    for q in sols:
        T = chain.fkine(np.asarray(q, float)).as_matrix()
        assert np.allclose(T[:2, 3], [1.0, 1.0], atol=1e-8)


def test_service_solve_problem_iterative_from_T():
    svc = InverseService()
    prob = {
        "model": {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
        "method": {"method": "iterative", "tol": 1e-9, "itmax": 250, "lambda": 1e-3},
        "pose": {"T": _T_xy(1.2, 0.3).tolist()},
    }
    sols = svc.solve_problem(prob)
    assert isinstance(sols, list) and len(sols) >= 1
    q = np.asarray(sols[0], float)
    chain = D.planar_2r(1.0, 1.0)
    T = chain.fkine(q).as_matrix()
    assert np.allclose(T[:2, 3], [1.2, 0.3], atol=1e-5)


def test_service_handles_unreachable_returns_empty_or_raises_clean():
    svc = InverseService()
    # Unreachable for l1=l2=1: x=3,y=0
    prob = {
        "model": {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
        "method": {"method": "analytic"},
        "pose": {"x": 3.0, "y": 0.0},
    }
    try:
        sols = svc.solve_problem(prob)
        assert isinstance(sols, list)
        assert len(sols) == 0
    except ValueError:
        # Acceptable behavior: explicit error for unreachable target
        pass
