import numpy as np
from angular.utils import skew, vee

def test_skew_vee_roundtrip():
    w = np.array([0.3, -0.2, 1.1])
    W = skew(w)
    assert np.allclose(vee(W), w)
    assert np.allclose(W.T, -W)
