import numpy as np
from path_planning.spatial import ParabolicBlend3D, Cycloid1D, Harmonic1D

def test_parabolic_blend_constant_accel_in_blend():
    pb = ParabolicBlend3D([0,0,0],[1,0,0],[1,1,0],0,1,2,0.1)
    t = np.linspace(0.9, 1.1, 10)
    acc = pb.sample(t).qdd
    # Use only interior points strictly inside the blend interval (exclude the boundaries)
    m = (t > 0.9) & (t < 1.1)
    # All interior rows should be equal to the first interior row
    interior = acc[m]
    assert interior.shape[0] > 0
    assert np.allclose(interior, interior[0], atol=1e-8)


def test_cycloid_rest_to_rest():
    cyc = Cycloid1D(0,1,0,1)
    assert np.isclose(cyc.qd(0), 0)
    assert np.isclose(cyc.qd(1), 0)

def test_harmonic_fit():
    harm = Harmonic1D.fit_rest2rest(0,1,10,45,np.pi)
    assert np.isfinite(harm.a0)
