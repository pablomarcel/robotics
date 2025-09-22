import sympy as sp
from applied.core import Inertia

def test_parallel_axis_matches_formula():
    m = sp.Symbol("m", positive=True)
    I = sp.diag(1,2,3)
    body = Inertia(m, I)
    r = sp.Matrix([1,0,0])
    shifted = body.parallel_axis(r)
    # New Ixx = 1 + m*(y^2+z^2) = 1 + 0
    assert sp.simplify(shifted.I[0,0] - 1) == 0
    # Iyy = 2 + m*(x^2+z^2) = 2 + m
    assert sp.simplify(shifted.I[1,1] - (2 + m)) == 0
