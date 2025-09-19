# motion/tests/test_screw_transform.py
import math
import numpy as np
import pytest

from motion.core import Rotation, SE3, Screw


def test_central_pure_rotation_about_z_90deg():
    s = Screw(u=np.array([0, 0, 1.0]), s=np.zeros(3), h=0.0, phi=math.pi / 2)
    T = s.to_matrix()
    p = np.array([1.0, 0.0, 0.0, 1.0])  # x-axis point
    out = T @ p
    assert np.allclose(out[:3], [0.0, 1.0, 0.0], atol=1e-12)
    # rotation block should equal Rz(90)
    Rz = Rotation.Rz(math.pi / 2).as_matrix()
    assert np.allclose(T[:3, :3], Rz, atol=1e-12)


def test_central_screw_with_pitch_translates_along_axis():
    h = 0.5
    phi = math.pi
    s = Screw(u=np.array([0.0, 0.0, 1.0]), s=np.zeros(3), h=h, phi=phi)
    T = s.to_matrix()
    # Translation part for a central screw is h * phi * u
    expected_t = h * phi * np.array([0.0, 0.0, 1.0])
    assert np.allclose(T[:3, 3], expected_t, atol=1e-12)
    # Apply to the origin
    out = (T @ np.array([0, 0, 0, 1.0]))[:3]
    assert np.allclose(out, expected_t, atol=1e-12)


def test_off_axis_rotation_matches_conjugation_D_R_Dinv():
    # Rotate about z-axis but through the point s = [2, 0, 0] (h = 0)
    s_vec = np.array([2.0, 0.0, 0.0])
    phi = math.radians(45.0)
    screw = Screw(u=np.array([0.0, 0.0, 1.0]), s=s_vec, h=0.0, phi=phi)
    T_screw = screw.to_matrix()

    # Construct via conjugation: D * R * D^{-1}
    D = SE3(np.eye(3), s_vec)
    R = SE3(Rotation.Rz(phi).as_matrix(), np.zeros(3))
    T_conj = (D @ R @ D.inv()).as_matrix()

    assert np.allclose(T_screw, T_conj, atol=1e-12)


def test_inverse_is_negating_angle_only():
    # General screw (non-zero s and h)
    u = np.array([1.0, 2.0, -1.0]); u /= np.linalg.norm(u)
    s_vec = np.array([0.3, -0.1, 0.7])
    h = 0.2
    phi = -0.8
    T = Screw(u=u, s=s_vec, h=h, phi=phi).to_matrix()
    T_inv_expected = np.linalg.inv(T)

    # Inverse screw: same axis & s, negate φ; pitch h is intrinsic to the twist and stays the same
    T_inv_by_params = Screw(u=u, s=s_vec, h=h, phi=-phi).to_matrix()

    assert np.allclose(T_inv_by_params, T_inv_expected, atol=1e-12)
    assert np.allclose(T @ T_inv_by_params, np.eye(4), atol=1e-12)
    assert np.allclose(T_inv_by_params @ T, np.eye(4), atol=1e-12)


def test_composition_same_axis_same_s_with_effective_pitch():
    # Compose two screws about the same line (same u, same s), different pitches.
    u = np.array([0.0, 0.0, 1.0])
    s_vec = np.array([1.5, -0.2, 0.3])
    phi1, h1 = 0.7, 0.10
    phi2, h2 = -0.4, 0.30

    T1 = Screw(u=u, s=s_vec, h=h1, phi=phi1).to_matrix()
    T2 = Screw(u=u, s=s_vec, h=h2, phi=phi2).to_matrix()
    T12 = T1 @ T2

    # For same twist axis and same s, composition is exp(S * (phi1+phi2)).
    # The resulting pitch satisfies h_eff * (phi1+phi2) = h1*phi1 + h2*phi2
    phi = phi1 + phi2
    # If phi is ~0, the decomposition is ill-defined; pick parameters so not zero.
    assert abs(phi) > 1e-9
    h_eff = (h1 * phi1 + h2 * phi2) / phi

    T_eff = Screw(u=u, s=s_vec, h=h_eff, phi=phi).to_matrix()
    assert np.allclose(T12, T_eff, atol=1e-12)


def test_small_angle_linearization_translation_term():
    # For small φ: R ≈ I + φ [u]_x, and t ≈ -(φ [u]_x) s + h φ u
    u = np.array([0.2, -0.4, 0.1]); u /= np.linalg.norm(u)
    s_vec = np.array([0.3, 0.5, -0.2])
    h = -0.25
    phi = 1e-7

    T = Screw(u=u, s=s_vec, h=h, phi=phi).to_matrix()

    # Linearized translation
    ux = np.array([[0, -u[2], u[1]],
                   [u[2], 0, -u[0]],
                   [-u[1], u[0], 0]], dtype=float)
    t_lin = -(phi * ux) @ s_vec + (h * phi) * u

    assert np.allclose(T[:3, :3], np.eye(3) + phi * ux, atol=1e-10)
    assert np.allclose(T[:3, 3], t_lin, atol=1e-10)
