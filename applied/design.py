# applied/design.py
"""
Preset builders for **applied dynamics** workflows (Eq. 10.1–10.398).

These presets return concrete **System** model instances from :mod:`applied.models`
with either **symbolic** or **numeric** parameters, ready for:
- Lagrange EOM derivation (:class:`applied.dynamics.LagrangeEngine`)
- Energy inspection (K, V)
- Mass matrix extraction (where applicable)

What you get
------------
* Dataclass **specs** describing each preset's parameters
* `build_*` functions that produce configured model instances
* A small **DesignLibrary** façade to list/create presets by name
* All builders are dependency-free (SymPy optional for numeric usage)

Examples
--------
>>> from applied.design import DesignLibrary
>>> lib = DesignLibrary()
>>> model = lib.create("pendulum_sym")           # SimplePendulum with (m,l,g) symbols
>>> names = lib.available()                      # discover keys

>>> from applied.apis import AppliedDynamicsAPI
>>> api = AppliedDynamicsAPI()
>>> res = api.derive_simple_pendulum()           # or build via library & derive yourself
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Callable

import sympy as sp

from .models import (
    SimplePendulum,
    SphericalPendulum,
    Planar2R,
    CartPendulumAbsorber,
)
from .core import System


# ------------------------------- Type alias -------------------------------- #

Scalar = Union[float, int, sp.Symbol]

# --------------------------------- Specs ----------------------------------- #

@dataclass(frozen=True)
class SimplePendulumSpec:
    m: Scalar = sp.symbols("m", positive=True)
    l: Scalar = sp.symbols("l", positive=True)
    g: Scalar = sp.symbols("g", positive=True)

    def build(self) -> SimplePendulum:
        return SimplePendulum(self.m, self.l, self.g)


@dataclass(frozen=True)
class SphericalPendulumSpec:
    m: Scalar = sp.symbols("m", positive=True)
    l: Scalar = sp.symbols("l", positive=True)
    g: Scalar = sp.symbols("g", positive=True)

    def build(self) -> SphericalPendulum:
        return SphericalPendulum(self.m, self.l, self.g)


@dataclass(frozen=True)
class Planar2RSpec:
    m1: Scalar = sp.symbols("m1", positive=True)
    m2: Scalar = sp.symbols("m2", positive=True)
    l1: Scalar = sp.symbols("l1", positive=True)
    l2: Scalar = sp.symbols("l2", positive=True)
    g: Scalar = sp.symbols("g", positive=True)

    def build(self) -> Planar2R:
        return Planar2R(self.m1, self.m2, self.l1, self.l2, self.g)


@dataclass(frozen=True)
class CartAbsorberSpec:
    M: Scalar = sp.symbols("M", positive=True)
    m: Scalar = sp.symbols("m", positive=True)
    l: Scalar = sp.symbols("l", positive=True)
    k: Scalar = sp.symbols("k", positive=True)
    g: Scalar = sp.symbols("g", positive=True)

    def build(self) -> CartPendulumAbsorber:
        return CartPendulumAbsorber(self.M, self.m, self.l, self.k, self.g)


# ------------------------------ Builders ----------------------------------- #

def build_pendulum(*, m: Scalar = None, l: Scalar = None, g: Scalar = None) -> SimplePendulum:
    """Simple pendulum with optional overrides (defaults symbolic)."""
    spec = SimplePendulumSpec(
        m if m is not None else SimplePendulumSpec().m,
        l if l is not None else SimplePendulumSpec().l,
        g if g is not None else SimplePendulumSpec().g,
    )
    return spec.build()


def build_spherical_pendulum(*, m: Scalar = None, l: Scalar = None, g: Scalar = None) -> SphericalPendulum:
    spec = SphericalPendulumSpec(
        m if m is not None else SphericalPendulumSpec().m,
        l if l is not None else SphericalPendulumSpec().l,
        g if g is not None else SphericalPendulumSpec().g,
    )
    return spec.build()


def build_planar_2r(
    *, m1: Scalar = None, m2: Scalar = None, l1: Scalar = None, l2: Scalar = None, g: Scalar = None
) -> Planar2R:
    spec = Planar2RSpec(
        m1 if m1 is not None else Planar2RSpec().m1,
        m2 if m2 is not None else Planar2RSpec().m2,
        l1 if l1 is not None else Planar2RSpec().l1,
        l2 if l2 is not None else Planar2RSpec().l2,
        g if g is not None else Planar2RSpec().g,
    )
    return spec.build()


def build_cart_absorber(
    *, M: Scalar = None, m: Scalar = None, l: Scalar = None, k: Scalar = None, g: Scalar = None
) -> CartPendulumAbsorber:
    spec = CartAbsorberSpec(
        M if M is not None else CartAbsorberSpec().M,
        m if m is not None else CartAbsorberSpec().m,
        l if l is not None else CartAbsorberSpec().l,
        k if k is not None else CartAbsorberSpec().k,
        g if g is not None else CartAbsorberSpec().g,
    )
    return spec.build()


# ------------------------------ Design Library ----------------------------- #

class DesignLibrary:
    """
    Small registry for ready-to-use dynamic **design presets**.

    Use :meth:`available` to list keys, and :meth:`create` to instantiate a model.

    Preset keys (initial set)
    -------------------------
    pendulum_sym, pendulum_num
    spherical_sym
    planar2r_sym, planar2r_num
    absorber_sym, absorber_num
    """

    def __init__(self):
        g = 9.81
        # Registry maps name -> zero-arg callable returning a System
        self._registry: Dict[str, Callable[[], System]] = {
            # symbolic
            "pendulum_sym": lambda: build_pendulum(),
            "spherical_sym": lambda: build_spherical_pendulum(),
            "planar2r_sym": lambda: build_planar_2r(),
            "absorber_sym": lambda: build_cart_absorber(),
            # numeric examples (lightweight defaults; tune in tests or CLI)
            "pendulum_num": lambda: build_pendulum(m=1.0, l=1.0, g=g),
            "planar2r_num": lambda: build_planar_2r(m1=1.0, m2=1.0, l1=1.0, l2=0.8, g=g),
            "absorber_num": lambda: build_cart_absorber(M=5.0, m=0.5, l=0.6, k=50.0, g=g),
        }

    def available(self) -> List[str]:
        """Return sorted preset names."""
        return sorted(self._registry.keys())

    def create(self, name: str) -> System:
        """Instantiate a preset by name."""
        try:
            return self._registry[name]()
        except KeyError as e:
            raise KeyError(f"Unknown design preset '{name}'. Available: {', '.join(self.available())}") from e

    # Extension hook for apps/tests
    def register(self, name: str, factory: Callable[[], System]) -> None:
        """Register a new preset."""
        if name in self._registry:
            raise ValueError(f"Preset '{name}' already exists")
        self._registry[name] = factory


# ------------------------------ Convenience -------------------------------- #

# Friendly aliases for external imports (mirrors inverse.design style)
build_simple_pendulum = build_pendulum
build_spherical = build_spherical_pendulum
build_2r = build_planar_2r
build_cart_pendulum_absorber = build_cart_absorber
