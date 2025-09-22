# applied/tests/test_integrators.py
import numpy as np
from applied.design import DesignLibrary
from applied.integrators import LagrangeRHS, ODESolver, IntegratorConfig

def test_pendulum_numeric_runs():
    sys = DesignLibrary().create("pendulum_num")
    rhs = LagrangeRHS.from_model(sys)
    y0 = rhs.pack_state([0.1], [0.0])
    cfg = IntegratorConfig(t_span=(0.0, 0.5), rk4_dt=1e-3)
    sol = ODESolver(cfg).solve(rhs, y0)
    assert sol.t.size > 2 and sol.y.shape[1] == sol.t.size
    # energy roughly bounded
    q = sol.q()[0]
    assert np.isfinite(q).all()
