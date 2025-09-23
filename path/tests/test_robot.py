import numpy as np
from path.robot import Planar2R

def test_2r_fk_ik_roundtrip():
    arm = Planar2R(0.25,0.25)
    th1 = np.deg2rad(30); th2 = np.deg2rad(60)
    x,y = arm.fk(th1, th2)
    th1b, th2b = arm.ik(np.array([x]), np.array([y]))
    assert np.allclose([th1, th2], [th1b[0], th2b[0]], atol=1e-6)
