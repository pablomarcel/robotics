# timeopt/tests/test_core_double_integrator.py
import pytest
from timeopt.app import MinTimeDoubleIntegrator

casadi = pytest.importorskip("casadi")

def test_double_integrator_sanity():
    prob = MinTimeDoubleIntegrator("di_test", x0=0.0, xf=1.0, m=1.0, F=10.0)
    res = prob.run()
    assert res.ok
    assert 0.01 <= res.data["tf"] <= 10.0
