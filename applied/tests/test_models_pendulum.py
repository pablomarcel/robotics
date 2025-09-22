import sympy as sp
from applied.models import SimplePendulum
from applied.core import FrameState

def test_pendulum_energy():
    m,l,g = sp.symbols('m l g', positive=True)
    model = SimplePendulum(m,l,g)
    t = sp.symbols('t')
    th = model.th(t)
    fs = FrameState(sp.Matrix([th]), sp.Matrix([sp.diff(th, t)]))
    K = model.kinetic(fs)
    V = model.potential(fs)
    assert K.has(sp.diff(th,t))
    assert V.has(sp.cos(th))
