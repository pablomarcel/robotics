import sympy as sp
from applied_dynamics.apis import AppliedDynamicsAPI

def test_pendulum_eom_shape_and_content():
    r = AppliedDynamicsAPI().derive_simple_pendulum()
    eom = r.data["EOM"]
    m,l,g,t = sp.symbols('m l g t')
    th = sp.Function('theta')
    # Expect form m l^2 θ̈ + m g l sin θ = 0
    theta = th(t)
    thetad = sp.diff(theta, t)
    thetadd = sp.diff(theta, t, 2)
    expected = sp.simplify(m*l**2*thetadd + m*g*l*sp.sin(theta))
    assert eom.shape == (1,1)
    assert sp.simplify(eom[0] - expected) == 0
