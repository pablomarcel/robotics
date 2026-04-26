import numpy as np
from control_techniques.app import ControlApp

def test_ct_pd_zero_error_zero_feedforward(app):
    r = app.planar2r()
    q = np.array([0.1, -0.2]); qd = np.zeros(2)
    out = app.api.robot_computed_torque(r, q, qd, q, np.zeros(2), np.zeros(2), wn=4.0, zeta=1.0)
    # With qd_d=qdd_d=0 and q_d=q, torque is just bias (gravity here since qd=0)
    tau = out["tau"]
    assert tau.shape == (2,)
