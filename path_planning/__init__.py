"""Path planning package covering equations 12.1..12.301.

Public entry points:
- path_planning.cli: CLI entry
- path_planning.app: high-level orchestration API
"""

from .core import (
    Trajectory1D, TrajectoryND, SampledTrajectory, BoundaryConditions,
    PathError,
)
from .time import LSPB, QuinticTime
from .poly import CubicPoly, QuinticPoly, SepticPoly, LeastSquaresPoly
from .segment import Piecewise1D
from .spatial import ParabolicBlend3D, Harmonic1D, Cycloid1D, ComposeYofX
from .robot import Planar2R
from .rotation import AngleAxisPath

__all__ = [
    "Trajectory1D","TrajectoryND","SampledTrajectory","BoundaryConditions","PathError",
    "LSPB","QuinticTime",
    "CubicPoly","QuinticPoly","SepticPoly","LeastSquaresPoly",
    "Piecewise1D",
    "ParabolicBlend3D","Harmonic1D","Cycloid1D","ComposeYofX",
    "Planar2R",
    "AngleAxisPath",
]
