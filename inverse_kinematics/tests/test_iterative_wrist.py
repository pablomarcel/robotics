# inverse_kinematics/tests/test_iterative_wrist.py
# Iterative IK orientation_kinematics-tracking tests on a 3-DOF spherical wrist.

from __future__ import annotations

import numpy as np
import pytest

from inverse_kinematics import design as D
from inverse_kinematics.core import IterativeIK
from inverse_kinematics.utils import rpy_to_R, homogeneous


def _orientation_only_target(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 4x4 pose with desired rotation_kinematics and zero translation."""
    R = rpy_to_R(roll, pitch, yaw)
    return homogeneous(R, np.zeros(3))


def test_iterative_tracks_orientation_spherical_wrist_type1():
    """
    For a pure 3-DOF spherical wrist, the iterative solver should match an
    arbitrary orientation_kinematics closely (no translation component).
    """
    wrist = D.spherical_wrist(wrist_type=1, d_tool=0.0)  # orientation_kinematics only
    Tdes = _orientation_only_target(0.25, -0.40, 0.15)
    q0 = np.zeros(3)

    solver = IterativeIK(lambda_damp=1e-3, tol=1e-9, itmax=300, space="body")
    q = solver.solve(wrist, Tdes, q0)[0]
    Tgot = wrist.fkine(q).as_matrix()

    # Orientation match
    assert np.allclose(Tgot[:3, :3], Tdes[:3, :3], atol=1e-4)
    # No translation requested; ensure it's near zero
    assert np.allclose(Tgot[:3, 3], 0.0, atol=1e-8)


@pytest.mark.parametrize(
    "rpy",
    [
        (0.10, 0.05, -0.25),
        (-0.30, 0.60, 0.20),
        (0.0, -0.9, 0.0),  # near a tougher pitch
    ],
)
def test_iterative_tracks_various_orientations(rpy):
    wrist = D.spherical_wrist(wrist_type=2, d_tool=0.0)
    Tdes = _orientation_only_target(*rpy)
    q0 = np.zeros(3)

    solver = IterativeIK(lambda_damp=2e-3, tol=5e-9, itmax=400, space="body")
    q = solver.solve(wrist, Tdes, q0)[0]
    Rgot = wrist.fkine(q).as_matrix()[:3, :3]

    assert np.allclose(Rgot, Tdes[:3, :3], atol=2e-4)
