import numpy as np
from robot.apis import DynamicsAPI
from robot.core import State


def test_sympy_engine_returns_mcg(planar2r_model, sample_state):
    q, qd, qdd = sample_state
    api = DynamicsAPI(engine="sympy")
    res = api.run(planar2r_model, State(q, qd, qdd), gravity=9.81)
    M = res["M"]; C = res["C"]; g = res["g"]; tau = res["tau"]
    assert M.shape == (2,2)
    assert g.shape == (2,)
    assert tau.shape == (2,)
    # Verify numeric identity M*qdd + C@qd + g ≈ tau
    lhs = M @ qdd + C @ qd + g
    assert np.allclose(lhs, tau, atol=1e-6)