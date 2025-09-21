# acceleration/tests/conftest.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

# Core backend/types
from acceleration.backends.base import ChainState
from acceleration.backends.numpy_backend import Planar2R


# ---------------------------------------------------------------------------
# Pytest configuration / markers
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register project-specific markers so pytest -q --strict-markers is happy."""
    config.addinivalue_line("markers", "pinocchio: tests requiring Pinocchio backend")
    config.addinivalue_line("markers", "drake: tests requiring Drake backend")
    config.addinivalue_line("markers", "slow: slow tests (can be deselected)")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Optional CLI toggles for test runs."""
    parser.addoption(
        "--allow-numpy-float-errors",
        action="store_true",
        default=False,
        help="Do not raise on NumPy floating-point warnings (default raises).",
    )


# ---------------------------------------------------------------------------
# Global test session setup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _numpy_determinism(pytestconfig: pytest.Config):
    """Make NumPy behavior predictable and strict for the whole session."""
    if not pytestconfig.getoption("--allow-numpy-float-errors"):
        np.seterr(all="raise")  # catch bugs early
    np.set_printoptions(precision=12, suppress=True, linewidth=120)
    yield


# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for tests needing random numbers."""
    return np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# Temp I/O dirs (mirror repo layout: acceleration/in, acceleration/out)
# ---------------------------------------------------------------------------

@pytest.fixture
def io_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Path]:
    """Create temporary in/out directories for tests that touch the filesystem."""
    in_dir = tmp_path / "acceleration" / "in"
    out_dir = tmp_path / "acceleration" / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return {"in": in_dir, "out": out_dir}


# ---------------------------------------------------------------------------
# Small numeric tolerances shared by tests
# ---------------------------------------------------------------------------

@pytest.fixture
def tol() -> Dict[str, float]:
    """Canonical absolute tolerances for vector/matrix comparisons."""
    return {"tiny": 1e-12, "strict": 1e-9, "loose": 5e-6}


# ---------------------------------------------------------------------------
# Ready-to-use 2R planar backend (no chain model indirection)
# ---------------------------------------------------------------------------

@pytest.fixture
def two_r_lengths() -> tuple[float, float]:
    """Link lengths used across tests."""
    return (0.7, 0.9)


@pytest.fixture
def two_r_backend(two_r_lengths: tuple[float, float]) -> Planar2R:
    """
    NumPy Planar2R backend instance.

    The only supported frame is "ee" (end-effector XY).
    """
    l1, l2 = two_r_lengths
    # _fd_check=False by default; set to True here if you want FD self-checks
    return Planar2R(l1=l1, l2=l2, _frame="ee", _fd_check=False)


# ---------------------------------------------------------------------------
# Optional dependencies presence checks & skip helpers
# ---------------------------------------------------------------------------

def _have_pinocchio() -> bool:
    try:
        import pinocchio  # noqa: F401
        return True
    except Exception:
        return False


def _have_drake() -> bool:
    try:
        import pydrake.common  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture
def require_pinocchio():
    """Skip the test if Pinocchio is not installed."""
    if not _have_pinocchio():
        pytest.skip("Pinocchio not available")


@pytest.fixture
def require_drake():
    """Skip the test if Drake is not installed."""
    if not _have_drake():
        pytest.skip("Drake (pydrake) not available")
