# time/tests/test_toppraline.py
import numpy as np
import pytest
from time_optimal_control.design import Planar2RGeom
from time_optimal_control.app import TwoRPathTimeScaler, TwoRParams

toppra = pytest.importorskip("toppra")

def test_toppraline_runs_and_returns_tf():
    geom = Planar2RGeom(l1=1.0, l2=1.0)
    qs = geom.path_line_y_const(y=0.5, x0=1.9, x1=0.5, n=150)
    prob = TwoRPathTimeScaler("twoR_line_test", qs, TwoRParams(tau_max=(100,100)))
    res = prob.run()
    assert res.ok
    assert res.data["tf"] > 0
    assert res.data["grid_size"] >= 2
