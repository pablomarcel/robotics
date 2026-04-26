from __future__ import annotations
from functools import wraps
from time import perf_counter
from typing import Callable, Any, Tuple


def timed(fn: Callable[..., Any]) -> Callable[..., Tuple[Any, float]]:
    """Decorator: time a function and return (result, seconds)."""
    @wraps(fn)
    def _w(*a, **k):
        t0 = perf_counter()
        r = fn(*a, **k)
        return r, perf_counter() - t0
    return _w