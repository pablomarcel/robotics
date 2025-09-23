import numpy as np
from path.spatial import ParabolicBlend3D, Cycloid1D, Harmonic1D

def test_parabolic_blend_constant_accel_in_blend():
    pb = ParabolicBlend3D([0,0,0],[1,0,0],[1,1,0],0,1,2,0.1)
    t = np.linspace(0.9,1.1,10)
    acc = pb.sample(t).qdd
    # acc constant along rows in blend interval
    assert np.allclose(acc, acc[0], atol=1e-8)

def test_cycloid_rest_to_rest():
    cyc = Cycloid1D(0,1,0,1)
    assert np.isclose(cyc.qd(0), 0)
    assert np.isclose(cyc.qd(1), 0)

def test_harmonic_fit():
    harm = Harmonic1D.fit_rest2rest(0,1,10,45,np.pi)
    assert np.isfinite(harm.a0)
