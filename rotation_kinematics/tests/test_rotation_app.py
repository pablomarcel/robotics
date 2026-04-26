# =============================
# File: rotation_kinematics/tests/test_rotation_app.py
# =============================
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from rotation_kinematics import rot_core as core

def _close(a, b, atol=1e-6):
    return np.allclose(a, b, atol=atol)

@pytest.mark.rot
def test_compose_decompose_zyx_30_20_10():
    ang_deg = [30, 20, 10]
    Robj = core.build_matrix("global", "zyx", ang_deg, degrees=True)
    M = Robj.as_matrix()
    M_ref = np.array([
        [ 0.813798, -0.469846,  0.342020],
        [ 0.543838,  0.823173, -0.163176],
        [-0.204874,  0.318796,  0.925417]
    ])
    assert _close(M, M_ref, 3e-6)
    a_back = core.decompose(Robj, "zyx", "global", degrees=True)
    assert _close(a_back, np.array(ang_deg), 1e-6)

@pytest.mark.rot
def test_repeat_12x15deg_about_Z():
    # 12 * 15° about Z => 180° about Z
    Robj = core.build_matrix("global", "zyx", [15, 0, 0], degrees=True)  # Z=15°, Y=0, X=0
    Rm = core.repeat_rotation(Robj, 12).as_matrix()
    M_ref = np.diag([-1.0, -1.0, 1.0])
    assert _close(Rm, M_ref, 1e-6)

@pytest.mark.rot
def test_transform_basis_rows_returns_Rt():
    Robj = core.build_matrix("global", "zyx", [30, 20, 10], degrees=True)
    P = np.eye(3)  # row-vectors e1,e2,e3
    Pg = core.transform_points(Robj, P)
    assert _close(Pg, Robj.as_matrix().T, 1e-6)

@pytest.mark.rot
def test_align_x_axis_right_handed_unit_det():
    u = [0.5, -0.2, 0.84]
    Robj = core.align_body_x(u)
    M = Robj.as_matrix()
    # determinant ~ +1, orthonormal
    assert np.isclose(np.linalg.det(M), 1.0, atol=1e-8)
    assert _close(M.T @ M, np.eye(3), 1e-8)
    # x-axis aligned
    x_body_in_global = M[:, 0]
    u_hat = np.array(u) / np.linalg.norm(u)
    assert _close(x_body_in_global, u_hat, 1e-8)

# --- Closed-form E(q) vs numeric Jacobian consistency -----------------

@pytest.mark.cf
def test_closed_form_E_matches_numeric_zyx_body_local():
    sympy = pytest.importorskip("sympy")
    from rotation_kinematics import rot_closedform as cf

    # Use radians everywhere for this check
    q = np.array([0.3, -0.5, 0.2], dtype=float)         # [a1, a2, a3]
    qdot = np.array([0.10, -0.20, 0.05], dtype=float)   # rad/s

    # Closed-form E for local/intrinsic ZYX, body-frame ω
    E_sym, (a1, a2, a3) = cf.E_matrix('zyx', convention='local', frame='body')
    E_num = np.array(E_sym.subs({a1: q[0], a2: q[1], a3: q[2]}), dtype=float)

    omega_sym = E_num @ qdot
    omega_num = core.angvel_from_rates('zyx', q, qdot, convention='local',
                                       degrees=False, frame='body')
    assert _close(omega_sym, omega_num, 5e-7)

@pytest.mark.cf
def test_closed_form_E_matches_numeric_zyz_body_local():
    sympy = pytest.importorskip("sympy")
    from rotation_kinematics import rot_closedform as cf

    q = np.array([0.1, 0.4, -0.2], dtype=float)         # radians
    qdot = np.array([0.05, -0.1, 0.02], dtype=float)    # rad/s

    E_sym, (a1, a2, a3) = cf.E_matrix('zyz', convention='local', frame='body')
    E_num = np.array(E_sym.subs({a1: q[0], a2: q[1], a3: q[2]}), dtype=float)

    omega_sym = E_num @ qdot
    omega_num = core.angvel_from_rates('zyz', q, qdot, convention='local',
                                       degrees=False, frame='body')
    assert _close(omega_sym, omega_num, 5e-7)

@pytest.mark.rot
def test_angvel_local_zyz_body_matches_closed_form_in_degrees():
    """
    Replace brittle numeric snapshot with a closed-form consistency check in degrees.
    """
    sympy = pytest.importorskip("sympy")
    from rotation_kinematics import rot_closedform as cf

    # Angles and rates in degrees/deg·s⁻¹
    q_deg   = np.array([10.0, 20.0, 30.0], dtype=float)
    qdot_deg = np.array([0.1, 0.2, 0.3], dtype=float)

    # Numeric ω from core (respects degrees=True)
    omega_num_deg = core.angvel_from_rates("zyz", q_deg, qdot_deg,
                                           convention="local", degrees=True, frame="body")

    # Closed-form E in radians, then convert to deg/s for comparison
    E_sym, (a1, a2, a3) = cf.E_matrix('zyz', convention='local', frame='body')
    subs = {a1: np.deg2rad(q_deg[0]),
            a2: np.deg2rad(q_deg[1]),
            a3: np.deg2rad(q_deg[2])}
    E_num = np.array(E_sym.subs(subs), dtype=float)

    omega_sym_rad = E_num @ np.deg2rad(qdot_deg)
    omega_sym_deg = np.rad2deg(omega_sym_rad)

    assert _close(omega_sym_deg, omega_num_deg, 5e-7)
