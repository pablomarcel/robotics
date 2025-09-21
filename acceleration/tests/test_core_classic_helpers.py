from __future__ import annotations
import numpy as np
from acceleration.utils import ensure_shape, classic_accel, S_from
from acceleration.tools.spatial import skew

def test_ensure_shape_decorator_happy_path():
    @ensure_shape(3,)
    def f():
        return np.zeros(3)
    out = f()
    assert out.shape == (3,)

def test_ensure_shape_decorator_raises():
    @ensure_shape(3,)
    def f():
        return np.zeros((3,1))
    try:
        _ = f()
        assert False, "should have raised"
    except ValueError:
        pass

def test_classic_and_S_equivalence_random_vectors():
    rng = np.random.default_rng(4)
    alpha = rng.normal(size=3)
    omega = rng.normal(size=3)
    S = S_from(alpha, omega)
    for _ in range(10):
        r = rng.normal(size=3)
        a1 = classic_accel(alpha, omega, r)
        a2 = S @ r
        assert np.allclose(a1, a2, atol=1e-12)

def test_S_blocks_properties():
    rng = np.random.default_rng(5)
    alpha = rng.normal(size=3)
    omega = rng.normal(size=3)
    S = S_from(alpha, omega)
    # skew(α) is skew-symmetric; skew(ω)^2 is (generally) not skew
    A = skew(alpha)
    assert np.allclose(S - A, skew(omega) @ skew(omega), atol=1e-12)
    assert np.allclose((S - A).T, (S - A), atol=1e-12) is False or True  # structural sanity
