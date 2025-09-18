"""
orientation.utils (OO version)
------------------------------
Object-oriented utilities for SO(3) math used across the orientation package.

Design
------
- NumericConfig: central place to configure tolerances, backend preferences.
- OrientationUtils: instance providing methods (normalize, skew, vex, vers,
  safe_acos, project_to_so3, expm_so3, levi_civita).
- A default singleton `UTILS` is exported, and thin module-level functions
  delegate to it for backward compatibility and convenience in tests.

This keeps code testable and mockable:
  - swap `UTILS` with a custom instance (e.g., different EPS)
  - inject a fake backend (disable SciPy, etc.)

Author: Robotics project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# Optional SciPy detection (kept private; accessed via OrientationUtils)
# --------------------------------------------------------------------------

def _detect_scipy_expm():
    try:
        from scipy.linalg import expm  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class NumericConfig:
    """Configuration for numeric utilities.

    Attributes
    ----------
    eps : float
        Tolerance used throughout (zero checks, comparisons).
    prefer_scipy : bool
        If True and SciPy is available, use SciPy's expm for exp_so3.
    """
    eps: float = 1e-12
    prefer_scipy: bool = True


# --------------------------------------------------------------------------
# Main OO utility
# --------------------------------------------------------------------------

class OrientationUtils:
    """OO facade for common SO(3) utility routines.

    The methods are instance methods to allow dependency injection and
    per-instance configuration during tests (e.g., custom EPS).
    """

    def __init__(self, config: Optional[NumericConfig] = None):
        self.config = config or NumericConfig()
        self._scipy_available = _detect_scipy_expm()
        self._scipy_expm = None
        if self._scipy_available and self.config.prefer_scipy:
            from scipy.linalg import expm as _expm  # type: ignore
            self._scipy_expm = _expm

    # ---------------- linear algebra & helpers ----------------

    @property
    def EPS(self) -> float:
        return self.config.eps

    def normalize(self, v: np.ndarray) -> np.ndarray:
        """Return v normalized; if ||v|| < eps, return v unchanged."""
        v = np.asarray(v, dtype=float).reshape(-1)
        n = float(np.linalg.norm(v))
        return v if n < self.EPS else v / n

    def skew(self, v: np.ndarray) -> np.ndarray:
        """Skew-symmetric (hat) operator for a 3-vector."""
        x, y, z = np.asarray(v, dtype=float).reshape(3)
        return np.array([[0.0, -z,  y],
                         [z,  0.0, -x],
                         [-y,  x,  0.0]])

    def vex(self, S: np.ndarray) -> np.ndarray:
        """Vee operator (inverse hat) for a 3×3 skew matrix."""
        S = np.asarray(S, dtype=float).reshape(3, 3)
        return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)

    def vers(self, phi: float) -> float:
        """Versine: 1 - cos(phi)."""
        return 1.0 - float(np.cos(phi))

    def safe_acos(self, x: float) -> float:
        """Clamp into [-1, 1] before arccos (returns radians in [0, π])."""
        return float(np.arccos(max(-1.0, min(1.0, x))))

    def project_to_so3(self, R: np.ndarray) -> np.ndarray:
        """Project a near-rotation to SO(3) via SVD (closest in Frobenius norm)."""
        R = np.asarray(R, dtype=float).reshape(3, 3)
        U, _, Vt = np.linalg.svd(R)
        Rproj = U @ Vt
        if np.linalg.det(Rproj) < 0:
            U[:, -1] *= -1
            Rproj = U @ Vt
        return Rproj

    def levi_civita(self) -> np.ndarray:
        """Return the Levi–Civita symbol ε_ijk as a 3×3×3 tensor."""
        eps = np.zeros((3, 3, 3))
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
        eps[2, 1, 0] = eps[0, 2, 1] = eps[1, 0, 2] = -1.0
        return eps

    # ---------------- Lie group exponential on so(3) ----------------

    def expm_so3(self, omega: np.ndarray) -> np.ndarray:
        """Compute exp(omega^) ∈ SO(3).

        If SciPy's expm is available and allowed, uses it; otherwise,
        falls back to the closed-form Rodrigues formula (stable and fast).
        """
        omega = np.asarray(omega, dtype=float).reshape(3)
        if self._scipy_expm is not None:
            return self._scipy_expm(self.skew(omega))  # type: ignore

        theta = float(np.linalg.norm(omega))
        if theta < self.EPS:
            # First-order approximation: I + omega^
            return np.eye(3) + self.skew(omega)

        u = omega / theta
        s = float(np.sin(theta))
        c = float(np.cos(theta))
        U = self.skew(u)
        # R = I c + (1-c) uu^T + s U
        return np.eye(3) * c + (1.0 - c) * np.outer(u, u) + s * U


# --------------------------------------------------------------------------
# Default singleton and thin module-level shims (backward compatibility)
# --------------------------------------------------------------------------

UTILS = OrientationUtils()
EPS = UTILS.EPS  # exported constant for convenience

def normalize(v: np.ndarray) -> np.ndarray:
    return UTILS.normalize(v)

def skew(v: np.ndarray) -> np.ndarray:
    return UTILS.skew(v)

def vex(S: np.ndarray) -> np.ndarray:
    return UTILS.vex(S)

def vers(phi: float) -> float:
    return UTILS.vers(phi)

def safe_acos(x: float) -> float:
    return UTILS.safe_acos(x)

def project_to_so3(R: np.ndarray) -> np.ndarray:
    return UTILS.project_to_so3(R)

def expm_so3(omega: np.ndarray) -> np.ndarray:
    return UTILS.expm_so3(omega)

def levi_civita() -> np.ndarray:
    return UTILS.levi_civita()
