import numpy as np
from path.rotation import AngleAxisPath

def test_angle_axis_path_endpoints():
    R0 = np.eye(3)
    Rf = np.diag([-1,-1,1])  # 180deg about Z
    path = AngleAxisPath(R0, Rf)
    Rstart = path.R([0])[0]; Rend = path.R([1])[0]
    assert np.allclose(Rstart, R0)
    assert np.allclose(Rend, Rf)
