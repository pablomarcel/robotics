# motion/tests/test_plucker_distance.py
import math
import numpy as np
import pytest

from motion.core import PluckerLine, SE3, Rotation


def _angle_and_distance(l1: PluckerLine, l2: PluckerLine):
    return l1.angle(l2), l1.distance(l2)


def test_angle_distance_between_skew_lines():
    # l1: x-axis through origin
    l1 = PluckerLine.from_points([0, 0, 0], [1, 0, 0])
    # l2: line through (0,1,1) toward (0,2,3) -> direction (0,1,2), which is ⟂ to x-axis
    l2 = PluckerLine.from_points([0, 1, 1], [0, 2, 3])

    alpha, d = _angle_and_distance(l1, l2)

    # They are skew (do not intersect) and orthogonal: angle = π/2, distance > 0 finite
    assert pytest.approx(alpha, rel=0, abs=1e-12) == math.pi / 2
    assert d > 0 and math.isfinite(d)


def test_intersecting_lines_distance_zero_and_reciprocal_zero():
    # Two axes crossing at origin, orthogonal
    l1 = PluckerLine.from_points([0, 0, 0], [1, 0, 0])  # x-axis
    l2 = PluckerLine.from_points([0, 0, 0], [0, 1, 0])  # y-axis

    # Distance must be zero, angle = 90 deg
    alpha = l1.angle(l2)
    d = l1.distance(l2)
    assert pytest.approx(alpha, rel=0, abs=1e-12) == math.pi / 2
    assert pytest.approx(d, rel=0, abs=1e-12) == 0.0

    # Reciprocal product should vanish for intersecting lines (4.388)
    assert pytest.approx(l2.reciprocal_product(l1), rel=0, abs=1e-12) == 0.0


def test_parallel_lines_distance_nan_and_manual_computation():
    # Parallel lines along x, offset by 0.3 in y
    l1 = PluckerLine.from_points([0, 0, 0], [1, 0, 0])
    l2 = PluckerLine.from_points([0, 0.3, 0], [1, 0.3, 0])

    alpha, d = _angle_and_distance(l1, l2)
    # Our implementation returns NaN for parallel lines (sin alpha ~ 0)
    assert pytest.approx(alpha, rel=0, abs=1e-12) == 0.0
    assert math.isnan(d)

    # Manual shortest distance between parallel lines equals offset magnitude
    expected = 0.3
    assert pytest.approx(expected, rel=0, abs=1e-12) == 0.3


def test_se3_transform_invariance_of_angle_and_distance():
    # Two skew lines
    l1 = PluckerLine.from_points([0, 0, 0], [1, 0, 0])        # x-axis
    l2 = PluckerLine.from_points([0, 1, 1], [0, 2, 3])

    alpha0, d0 = _angle_and_distance(l1, l2)

    # Apply the same rigid transform to both lines
    T = SE3.from_rt(Rotation.Rz(np.deg2rad(30)), [0.5, -0.2, 0.3])
    l1p = l1.transform(T)
    l2p = l2.transform(T)

    alpha1, d1 = _angle_and_distance(l1p, l2p)

    # Angle and distance between lines are invariant under SE(3)
    assert pytest.approx(alpha1, rel=0, abs=1e-12) == alpha0
    assert pytest.approx(d1, rel=0, abs=1e-12) == d0
