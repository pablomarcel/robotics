# velocity/tests/test_solvers.py
"""
Solver-focused tests:
- Weighted damped least squares matches explicit weighting construction
- Adaptive damping path (damping=None) returns finite solution with small residual
- Orientation-only Newton IK on a 3R spherical wrist (ZXZ target) converges
"""

from __future__ import annotations

import numpy as np

from velocity import core, design


# --------------------------------------------------------------------------- #
# Weighted damped least squares
# --------------------------------------------------------------------------- #

def test_resolved_rates_weighted_matches_explicit():
    # Random but reproducible small problem (m=4 task dims, n=3 joints)
    rng = np.random.default_rng(123)
    J = rng.standard_normal((4, 3))
    xdot = rng.standard_normal(4)
    weights = np.array([1.0, 0.5, 2.0, 0.25])  # diagonal task weights
    lam = 1e-3

    # Solver path
    q_solver = core.solvers.resolved_rates(J, xdot, damping=lam, weights=weights)

    # Explicit construction: W^{1/2} J, W^{1/2} x
    Wsqrt = np.diag(np.sqrt(weights))
    Jw = Wsqrt @ J
    xw = Wsqrt @ xdot
    q_explicit = Jw.T @ np.linalg.inv(Jw @ Jw.T + (lam**2) * np.eye(Jw.shape[0])) @ xw

    assert np.allclose(q_solver, q_explicit, atol=1e-12)


# --------------------------------------------------------------------------- #
# Adaptive damping path (damping=None)
# --------------------------------------------------------------------------- #

def test_resolved_rates_adaptive_damping_is_finite_and_small_residual():
    # Use a nearly singular Jacobian: two nearly collinear columns
    e1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e1p = e1 + 1e-8 * np.array([0, 1, 0, 0, 0, 0], float)
    J = np.column_stack([e1, e1p, np.array([0, 1, 0, 0, 0, 0])])
    xdot = np.array([0.1, 0.05, 0.0, 0, 0, 0], float)

    q = core.solvers.resolved_rates(J, xdot, damping=None, weights=None)  # triggers adaptive λ

    assert np.all(np.isfinite(q))
    residual = np.linalg.norm(J @ q - xdot)
    # Should be small but not necessarily machine-precision due to ill-conditioning
    assert residual <= 1e-6 + 1e-2 * np.linalg.norm(xdot)


# --------------------------------------------------------------------------- #
# Orientation-only IK on a spherical wrist (ZXZ)
# --------------------------------------------------------------------------- #

def test_newton_ik_orientation_only_on_wrist_converges():
    # 3R spherical wrist; TCP at wrist center to isolate orientation
    wrist = design.spherical_wrist(wrist_type=1, d_tool=0.0)
    q0 = np.array([0.1, -0.2, 0.3])

    # Target orientation: ZXZ Euler (avoid singular beta≈0 or π)
    target_angles = np.array([0.7, 1.0, -0.8])  # radians
    x_tgt = {"p": [0.0, 0.0, 0.0], "euler": {"seq": "ZXZ", "angles": target_angles}}

    q_sol, info = core.solvers.newton_ik(
        wrist,
        q0,
        x_target=x_tgt,
        max_iter=120,
        tol=1e-10,
        weights=None,
        euler="ZXZ",
    )

    # Check convergence
    assert info["converged"] is True

    # Verify achieved orientation matches the target (position irrelevant for wrist-only chain)
    R_sol = wrist.fk(q_sol)["T_0e"][:3, :3]
    # Reconstruct target R for ZXZ:
    alpha, beta, gamma = target_angles
    # Build ZXZ rotation
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz_a = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Rx_b = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    Rz_g = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    R_tgt = Rz_a @ Rx_b @ Rz_g

    # Orientation error (angle-axis magnitude)
    Re = R_sol.T @ R_tgt
    ang_err = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1.0, 1.0))
    assert ang_err < 1e-6
