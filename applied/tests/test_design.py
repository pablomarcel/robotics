# applied/tests/test_design.py
from __future__ import annotations

import inspect
from typing import Iterable, Tuple, Any

import numpy as np
import pytest

try:
    import sympy as sp  # symbolic checks are optional in case sympy is vendored
except Exception:  # pragma: no cover - if sympy isn't present the sym tests will skip
    sp = None  # type: ignore

from applied.design import DesignLibrary


# ------------------------------ helpers ------------------------------------ #

def _to_int(x: Any) -> int:
    """Return int whether `x` is an int or a zero-arg callable producing an int."""
    if isinstance(x, int):
        return x
    if callable(x):
        sig = inspect.signature(x)
        if len(sig.parameters) == 0:
            return int(x())
    raise AssertionError(f"Cannot coerce {x!r} to int; expected dof int or zero-arg callable")

def _len_or_shape1(x: Any) -> int:
    """Length of 1-D iterable/array; for numpy arrays, use shape[0]."""
    if hasattr(x, "shape"):
        return int(x.shape[0])
    try:
        return len(x)
    except Exception:
        raise AssertionError(f"Object {type(x).__name__} has no length/shape")

def _zeros_like_coords(coords: Any) -> np.ndarray:
    """Create a numeric zero vector compatible with `coords`."""
    n = _len_or_shape1(coords)
    return np.zeros(n, dtype=float)

def _call_state_fn(fn, q, qd):
    """
    Call a state function with flexible signatures:
    - f()                        # no args
    - f(q) or f(qd)              # 1 arg
    - f(q, qd) / f(q, qd, *_)    # 2+ args (we pass q, qd and ignore extras via defaults)
    """
    sig = inspect.signature(fn)
    nreq = sum(1 for p in sig.parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
    nargs = len(sig.parameters)
    if nreq == 0:
        return fn()
    if nargs >= 2:
        return fn(q, qd)  # typical (q, qd, ...)
    if nargs == 1:
        # Try q first; if it errors, try qd
        try:
            return fn(q)
        except Exception:
            return fn(qd)
    # Fallback: attempt plain call
    return fn(q, qd)

def _is_sym_vector(vec: Iterable[Any]) -> bool:
    if sp is None:
        return False
    return all(isinstance(v, sp.Basic) for v in vec)

# ------------------------------- tests ------------------------------------- #

def test_library_has_presets_and_unique_names():
    lib = DesignLibrary()
    names = lib.available()
    assert isinstance(names, (list, tuple)) and names, "expected at least one preset"
    assert len(names) == len(set(names)), "preset names must be unique"

@pytest.mark.parametrize("name", DesignLibrary().available())
def test_preset_instantiation_and_shape(name):
    lib = DesignLibrary()
    sys = lib.create(name)

    # --- dof & coordinates
    dof_attr = getattr(sys, "dof", None)
    assert dof_attr is not None, f"{name}: system exposes 'dof' (int or zero-arg callable)"
    dof = _to_int(dof_attr)
    assert dof >= 1, f"{name}: dof must be >= 1"

    q = getattr(sys, "q", None)
    qd = getattr(sys, "qd", None)
    assert q is not None and qd is not None, f"{name}: must expose q and qd"
    assert _len_or_shape1(q) == dof, f"{name}: len(q) must equal dof"
    assert _len_or_shape1(qd) == dof, f"{name}: len(qd) must equal dof"

    # --- lagrangian callable (either name)
    lag = getattr(sys, "lagrangian", None) or getattr(sys, "lagrangian_state", None)
    assert callable(lag), f"{name}: must expose lagrangian() or lagrangian_state()"

    # --- energy accessor (returns K, V or dict)
    energy = getattr(sys, "energy", None)
    assert callable(energy), f"{name}: must expose energy()"

    # --- optional EOM
    eom = getattr(sys, "equations_of_motion", None)
    if eom is not None:
        assert callable(eom), f"{name}: equations_of_motion must be callable if present"

@pytest.mark.parametrize("name", [n for n in DesignLibrary().available() if n.endswith("_num")])
def test_numeric_presets_evaluate_lagrangian_energy_and_eom_at_zero(name):
    lib = DesignLibrary()
    sys = lib.create(name)

    q = getattr(sys, "q")
    qd = getattr(sys, "qd")
    q0 = _zeros_like_coords(q)
    qd0 = _zeros_like_coords(qd)

    # Lagrangian should be finite number
    lag = getattr(sys, "lagrangian", None) or getattr(sys, "lagrangian_state", None)
    L = _call_state_fn(lag, q0, qd0)
    # allow scalar, numpy scalar, or dict with "L"
    if isinstance(L, dict) and "L" in L:
        L = L["L"]
    Lf = float(np.asarray(L).squeeze())
    assert np.isfinite(Lf), f"{name}: Lagrangian must be finite at zero state"

    # Energy should produce K, V (numbers) or a mapping with 'K'/'V'
    E = sys.energy(q0, qd0) if len(inspect.signature(sys.energy).parameters) >= 2 else sys.energy()
    if isinstance(E, dict):
        K, V = E.get("K"), E.get("V")
    elif isinstance(E, tuple) and len(E) == 2:
        K, V = E
    else:
        raise AssertionError(f"{name}: energy() must return (K,V) or a dict with 'K' and 'V'")
    Kf, Vf = float(np.asarray(K).squeeze()), float(np.asarray(V).squeeze())
    assert np.isfinite(Kf) and np.isfinite(Vf), f"{name}: energies must be finite at zero state"

    # If EOM exists, it should return a vector (shape dof) at zero state
    eom = getattr(sys, "equations_of_motion", None)
    if callable(eom):
        tau = eom(q0, qd0) if len(inspect.signature(eom).parameters) >= 2 else eom()
        arr = np.atleast_1d(np.asarray(tau, dtype=float))
        assert arr.shape[0] == _to_int(getattr(sys, "dof")), f"{name}: EOM length == dof"
        assert np.all(np.isfinite(arr)), f"{name}: EOM must be finite at zero state"

@pytest.mark.skipif(sp is None, reason="sympy not available")
@pytest.mark.parametrize("name", [n for n in DesignLibrary().available() if n.endswith("_sym")])
def test_symbolic_presets_coordinates_are_sympy(name):
    lib = DesignLibrary()
    sys = lib.create(name)
    assert _is_sym_vector(sys.q), f"{name}: q must be SymPy symbols/expressions"
    assert _is_sym_vector(sys.qd), f"{name}: qd must be SymPy symbols/expressions"
