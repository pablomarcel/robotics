# applied/backends/base.py
from __future__ import annotations

"""
Common interfaces and utilities for numeric backends (Drake, Pinocchio, RTB).

What you get
------------
- Error taxonomy:
    * BackendNotAvailable   → library not installed or feature missing
    * BackendOpError        → runtime error surfaced from the backend

- Data carriers:
    * DynamicsResult        → {M, bias, K, V, extra} + helpers

- Interfaces:
    * DynamicsBackend (Protocol)        → minimal required surface
    * AbstractDynamicsBackend (base)    → default 'not installed' stubs

- Registry & selection:
    * BackendRegistry.register(cls)
    * BackendRegistry.available()
    * select_backend(preferred=None)    → instantiate first available

- NumPy helpers (shape/typing consistency):
    * as1d(x, n=None), as2d(x, m=None, n=None)
    * assert_shape(A, *shape)
    * split_state(x) ↔ stack_state(q, qd)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Type, Union

import numpy as np


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class BackendNotAvailable(RuntimeError):
    """Raised when a requested backend/library or a specific feature is not installed/available."""


class BackendOpError(RuntimeError):
    """Raised when a backend operation fails at runtime (bad inputs, internal error, etc.)."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise BackendNotAvailable(msg)


# ---------------------------------------------------------------------------
# Typing aliases
# ---------------------------------------------------------------------------

Number = Union[int, float]
NDArray = np.ndarray


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class DynamicsResult:
    """
    Generic numeric result for verification against symbolic dynamics.

    Attributes
    ----------
    M    : (n,n) mass/inertia matrix at q
    bias : (n,) Coriolis+gravity generalized force vector C(q,qd)@qd + g(q)
    K    : float kinetic energy
    V    : float potential energy
    extra: dict backend-specific extras (frames, diagnostics, etc.)
    """
    M: Optional[NDArray] = None
    bias: Optional[NDArray] = None
    K: Optional[float] = None
    V: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def empty() -> "DynamicsResult":
        return DynamicsResult()

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe-ish view (lists instead of ndarrays when present)."""
        def conv(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x
        return {
            "M": conv(self.M),
            "bias": conv(self.bias),
            "K": conv(self.K),
            "V": conv(self.V),
            "extra": {k: conv(v) for k, v in (self.extra or {}).items()},
        }


# ---------------------------------------------------------------------------
# Backend protocol (minimal interface each backend must implement)
# ---------------------------------------------------------------------------

class DynamicsBackend(Protocol):
    """
    Minimal interface a backend must implement. Methods can raise
    BackendNotAvailable if a capability isn't present for that backend.
    """
    name: str

    # --- discovery ---
    @staticmethod
    def available() -> bool: ...

    # --- builders for canonical systems ---
    def build_simple_pendulum(self, m: Number, l: Number, g: Number) -> Any: ...
    def build_planar_2r(self, m1: Number, m2: Number, l1: Number, l2: Number, g: Number) -> Any: ...
    def build_cart_pendulum_absorber(self, M: Number, m: Number, l: Number, k: Number, g: Number) -> Any: ...

    # --- numeric queries ---
    def mass_matrix(self, model: Any, q: NDArray) -> NDArray: ...
    def bias_coriolis_gravity(self, model: Any, q: NDArray, qd: NDArray) -> NDArray: ...
    def energies(self, model: Any, q: NDArray, qd: NDArray) -> Tuple[float, float]: ...

    # --- tiny sim (optional) ---
    def step(self, model: Any, q: NDArray, qd: NDArray, tau: NDArray, dt: float) -> Tuple[NDArray, NDArray]: ...


# ---------------------------------------------------------------------------
# Abstract base with graceful defaults
# ---------------------------------------------------------------------------

class AbstractDynamicsBackend:
    """
    Convenience base class that surfaces a *consistent* error if the backend
    isn’t installed. Inherit from this and override what you support.
    """
    name: str = "abstract"

    @staticmethod
    def available() -> bool:
        return False

    # ---- builders ----
    def build_simple_pendulum(self, *a, **k):  # pragma: no cover - default path
        raise BackendNotAvailable(f"{self.name}: simple pendulum builder is unavailable (backend not installed).")

    def build_planar_2r(self, *a, **k):  # pragma: no cover
        raise BackendNotAvailable(f"{self.name}: planar_2r builder is unavailable (backend not installed).")

    def build_cart_pendulum_absorber(self, *a, **k):  # pragma: no cover
        raise BackendNotAvailable(f"{self.name}: cart-pendulum absorber builder is unavailable (backend not installed).")

    # ---- numeric ----
    def mass_matrix(self, *a, **k) -> NDArray:  # pragma: no cover
        raise BackendNotAvailable(f"{self.name}: mass_matrix is unavailable (backend not installed).")

    def bias_coriolis_gravity(self, *a, **k) -> NDArray:  # pragma: no cover
        raise BackendNotAvailable(f"{self.name}: bias (Cqd+g) is unavailable (backend not installed).")

    def energies(self, *a, **k) -> Tuple[float, float]:  # pragma: no cover
        raise BackendNotAvailable(f"{self.name}: energies is unavailable (backend not installed).")

    def step(self, *a, **k) -> Tuple[NDArray, NDArray]:  # pragma: no cover
        raise BackendNotAvailable(f"{self.name}: step() is unavailable (backend not installed).")


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

class BackendRegistry:
    """
    Registry of backend classes. Your backend module should call:

        BackendRegistry.register(MyDrakeBackend)

    Then higher-level code can do:

        be = select_backend(preferred=["drake", "pinocchio"])
    """
    _classes: List[Type[DynamicsBackend]] = []

    @classmethod
    def register(cls, backend_cls: Type[DynamicsBackend]) -> None:
        if backend_cls not in cls._classes:
            cls._classes.append(backend_cls)

    @classmethod
    def classes(cls) -> List[Type[DynamicsBackend]]:
        return list(cls._classes)

    @classmethod
    def available(cls) -> List[str]:
        names: List[str] = []
        for klass in cls._classes:
            try:
                if klass.available():
                    # try to read a friendly name (fallback to class name)
                    nm = getattr(klass, "name", klass.__name__).lower()
                    names.append(nm)
            except Exception:
                # Swallow import errors here; selection will surface cleanly.
                continue
        return names


def select_backend(preferred: Optional[Sequence[str]] = None) -> DynamicsBackend:
    """
    Instantiate the first available backend, honoring an optional priority list.

    Parameters
    ----------
    preferred : list[str] | None
        Names in order of preference (e.g., ["drake", "pinocchio", "rtb"]).
        Names are matched case-insensitively against the backend class `.name`
        or the class name.

    Returns
    -------
    DynamicsBackend
        An initialized backend instance.

    Raises
    ------
    BackendNotAvailable
        If no registered backend is available.
    """
    # Normalize preference list
    pref = [p.lower() for p in preferred] if preferred else []

    # Build candidate class list honoring preference order
    candidates: List[Type[DynamicsBackend]] = BackendRegistry.classes()
    if pref:
        # stable sort by index in pref (unknowns go to the end)
        def order_key(klass: Type[DynamicsBackend]) -> Tuple[int, str]:
            nm = getattr(klass, "name", klass.__name__).lower()
            try:
                return (pref.index(nm), nm)
            except ValueError:
                return (len(pref) + 1, nm)
        candidates = sorted(candidates, key=order_key)

    # Pick the first available
    for klass in candidates:
        try:
            if klass.available():
                return klass()  # type: ignore[call-arg]
        except Exception:
            continue

    raise BackendNotAvailable("No numeric backend is available. Install one (e.g., Drake or Pinocchio) "
                              "and ensure its backend module registers with BackendRegistry.register(...).")


# ---------------------------------------------------------------------------
# NumPy helpers (shape & coercion)
# ---------------------------------------------------------------------------

def as1d(x: Any, n: Optional[int] = None, *, dtype=float) -> NDArray:
    """Coerce to a 1D ndarray; optionally check length n."""
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    if n is not None and arr.size != n:
        raise ValueError(f"Expected vector of length {n}, got shape {arr.shape}")
    return arr


def as2d(x: Any, m: Optional[int] = None, n: Optional[int] = None, *, dtype=float) -> NDArray:
    """Coerce to a 2D ndarray; optionally check shape (m,n)."""
    A = np.asarray(x, dtype=dtype)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {A.shape}")
    if m is not None and A.shape[0] != m:
        raise ValueError(f"Expected {m} rows, got {A.shape[0]}")
    if n is not None and A.shape[1] != n:
        raise ValueError(f"Expected {n} cols, got {A.shape[1]}")
    return A


def assert_shape(A: NDArray, *shape: int) -> None:
    """Assert array has the exact shape (raises ValueError on mismatch)."""
    if tuple(A.shape) != tuple(shape):
        raise ValueError(f"Expected shape {shape}, got {A.shape}")


def stack_state(q: NDArray, qd: NDArray) -> NDArray:
    """Concatenate [q, qd] into a 1D state vector."""
    return np.r_[as1d(q), as1d(qd)]


def split_state(x: NDArray) -> Tuple[NDArray, NDArray]:
    """Split a 1D state vector into (q, qd) halves (assumes even length)."""
    x = as1d(x)
    _require(x.size % 2 == 0, "State vector must have even length: [q, qd].")
    n = x.size // 2
    return x[:n].copy(), x[n:].copy()
