# acceleration/tests/test_design.py
from __future__ import annotations
import numpy as np

from acceleration.backends.numpy_backend import Planar2R
from acceleration.backends.base import ChainState


def test_planar2r_preset_forward_accel_smoke():
    # Build the simple NumPy backend for a 2R planar arm (XY task)
    l1, l2 = 0.7, 0.9
    be = Planar2R(l1=l1, l2=l2)

    rng = np.random.default_rng(7)
    q   = rng.uniform(low=-0.8, high=0.8, size=2)
    qd  = rng.normal(size=2) * 0.5
    qdd = rng.normal(size=2) * 0.3

    st = ChainState(q=q, qd=qd, qdd=qdd)
    a_xy = be.spatial_accel("ee", st)

    # XY task → 2-vector
    assert a_xy.shape == (2,)
    assert a_xy.dtype == float
