import numpy as np
from control_techniques.app import ControlApp

def test_linearize_matches_signs_and_dims(app):
    pl = app.pendulum()
    op_x = np.array([np.pi/2, 0.0]); op_u = np.array([pl.m*pl.g*pl.l])
    L = app.api.linearize_pendulum(pl, op_x, op_u)
    A,B = L["A"], L["B"]
    assert A.shape == (2,2) and B.shape == (2,1)
    # ∂θdot/∂θ = 0, ∂θdot/∂θdot = 1 near operating point
    assert abs(A[0,0]) < 1e-7 and abs(A[0,1]-1.0) < 1e-7

def test_pendulum_pid_runs_and_returns(app):
    res = app.run_pendulum_pid_at_pi_over_2(T=0.2)
    assert res.x.shape[1] == 2
