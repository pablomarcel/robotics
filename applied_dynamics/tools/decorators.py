# applied_dynamics/tools/decorators.py
from __future__ import annotations

"""
Utility decorators for file-emitting helpers in the **applied_dynamics** module.

What you get
------------
- ensure_outfile_param(param="outfile", default=None)
    Ensure a path parameter exists (kwarg) and parent dirs are created.
    * `default` may be:
        - a string path,
        - a format string (e.g., "applied_dynamics/out/{name}.txt"),
        - a callable: default(*args, **kwargs) -> str | Path | None.

- write_text_result(param="outfile", encoding="utf-8", atomic=True)
    If the wrapped function returns `str`, write it to the outfile path.
    Returns the output `Path`.

- write_json_result(param="outfile", indent=2, default=str, atomic=True)
    If the wrapped function returns a Mapping or Sequence, JSON-dumps it.

- export_text(default=None, param="outfile", encoding="utf-8")
    Shorthand = ensure_outfile_param + write_text_result.

- export_json(default=None, param="outfile", indent=2, default_encoder=str)
    Shorthand = ensure_outfile_param + write_json_result.

- writes_to(out_path)  (back-compat)
    Older decorator that ensures an 'outfile' kwarg; now powered by
    `ensure_outfile_param`. Kept for callers/tests relying on it.

All decorators are **pure-Python**, no heavy deps, and are easy to unit test.
"""

from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Optional, Union, Tuple
import json
import os
import tempfile

from ..utils import ensure_outfile

PathLike = Union[str, Path]
Resolver = Callable[..., Optional[PathLike]]

__all__ = [
    "ensure_outfile_param",
    "write_text_result",
    "write_json_result",
    "export_text",
    "export_json",
    "writes_to",            # back-compat
]

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve_default(default: Optional[Union[PathLike, str, Resolver]],
                     args: Tuple[Any, ...],
                     kwargs: Mapping[str, Any]) -> Optional[Path]:
    """
    Resolve a default outfile specification into a Path (or None).

    - If default is callable: call(default)(*args, **kwargs)
    - If default is a string containing '{...}': format with kwargs
    - Else treat default as a literal path
    """
    if default is None:
        return None
    if callable(default):
        val = default(*args, **kwargs)
        return None if val is None else Path(val)
    # format-style template using kwargs
    if isinstance(default, str) and "{" in default and "}" in default:
        try:
            formatted = default.format(**kwargs)
        except Exception:
            formatted = default  # fall back if some field missing
        return Path(formatted)
    return Path(default)


def _coerce_outfile(maybe: Any) -> Optional[Path]:
    if isinstance(maybe, (str, Path)):
        return Path(maybe)
    return None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    Write bytes atomically (best-effort) by writing to a temp file and renaming.
    Ensures parent exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), prefix=".tmp_", suffix=".part") as tmp:
        tmp.write(data)
        temp_name = tmp.name
    # On POSIX, replace is atomic; on Windows, os.replace works similarly for files
    os.replace(temp_name, str(path))


# ---------------------------------------------------------------------------
# Public decorators
# ---------------------------------------------------------------------------

def ensure_outfile_param(
    *,
    param: str = "outfile",
    default: Optional[Union[PathLike, str, Resolver]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Ensure a function receives a writable output path in a kwarg.

    Behavior
    --------
    - If the kwarg `param` is provided (str|Path), parents are created.
    - Else, try to resolve `default` (callable or format string is allowed).
    - Else, if the last positional argument *looks* like a path, use that.
    - The resolved path is injected back into kwargs[param] as a `Path`.
    - Returns the wrapped function's normal return value.

    This decorator **does not** perform the write; pair it with
    `write_text_result` or `write_json_result` if you want the decorator to
    perform I/O based on the function's return value.
    """
    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = _coerce_outfile(kwargs.get(param))
            if out is None and args:
                out = _coerce_outfile(args[-1])  # conventional last positional path
            if out is None:
                out = _resolve_default(default, args, kwargs)
            if out is None:
                raise ValueError(f"Missing output path for parameter '{param}' and no usable default.")
            ensure_outfile(out)  # creates parents
            kwargs[param] = out
            return func(*args, **kwargs)
        return wrapped
    return deco


def write_text_result(
    *,
    param: str = "outfile",
    encoding: str = "utf-8",
    atomic: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Path]]:
    """
    After the wrapped function returns:
      - If the result is a `str`, write it to kwargs[param] and return the `Path`.
      - If it's a `(str, meta)` tuple, the first element is written; meta is ignored here.
      - Otherwise, pass the result through unchanged.

    Typical usage: stack after `ensure_outfile_param`.
    """
    def deco(func: Callable[..., Any]) -> Callable[..., Path]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            res = func(*args, **kwargs)
            out = _coerce_outfile(kwargs.get(param))
            if out is None:
                return res
            text: Optional[str] = None
            if isinstance(res, str):
                text = res
            elif isinstance(res, tuple) and res and isinstance(res[0], str):
                text = res[0]
            if text is None:
                return res  # nothing to write; transparent behavior
            data = text.encode(encoding)
            if atomic:
                _atomic_write_bytes(out, data)
            else:
                out.write_text(text, encoding=encoding)
            return out
        return wrapped
    return deco


def write_json_result(
    *,
    param: str = "outfile",
    indent: int = 2,
    default: Callable[[Any], Any] = str,
    atomic: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Path]]:
    """
    After the wrapped function returns:
      - If the result is Mapping/Sequence, JSON-dump to kwargs[param] and return `Path`.
      - If it's a `(payload, meta)` tuple and `payload` is Mapping/Sequence, dump payload.
      - Otherwise, pass result through.
    """
    def deco(func: Callable[..., Any]) -> Callable[..., Path]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            res = func(*args, **kwargs)
            out = _coerce_outfile(kwargs.get(param))
            if out is None:
                return res
            payload: Optional[Any] = None
            if isinstance(res, (Mapping, Sequence)) and not isinstance(res, (str, bytes, bytearray)):
                payload = res
            elif isinstance(res, tuple) and res and isinstance(res[0], (Mapping, Sequence)):
                payload = res[0]
            if payload is None:
                return res
            data = json.dumps(payload, indent=indent, default=default).encode("utf-8")
            if atomic:
                _atomic_write_bytes(out, data)
            else:
                out.write_text(json.dumps(payload, indent=indent, default=default), encoding="utf-8")
            return out
        return wrapped
    return deco


# -------------------------- One-shot convenience ---------------------------

def export_text(
    default: Optional[Union[PathLike, str, Resolver]] = None,
    *,
    param: str = "outfile",
    encoding: str = "utf-8",
    atomic: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Path]]:
    """
    Shorthand for: ensure_outfile_param(..., default) + write_text_result(...)
    """
    def chain(func: Callable[..., Any]) -> Callable[..., Path]:
        return ensure_outfile_param(param=param, default=default)(
            write_text_result(param=param, encoding=encoding, atomic=atomic)(func)
        )
    return chain


def export_json(
    default: Optional[Union[PathLike, str, Resolver]] = None,
    *,
    param: str = "outfile",
    indent: int = 2,
    default_encoder: Callable[[Any], Any] = str,
    atomic: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Path]]:
    """
    Shorthand for: ensure_outfile_param(..., default) + write_json_result(...)
    """
    def chain(func: Callable[..., Any]) -> Callable[..., Path]:
        return ensure_outfile_param(param=param, default=default)(
            write_json_result(param=param, indent=indent, default=default_encoder, atomic=atomic)(func)
        )
    return chain


# --------------------------- Back-compat wrapper ---------------------------

def writes_to(out_path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Backwards-compatible decorator used in earlier code.
    It ensures an 'outfile' kwarg exists (creating parents if needed).
    """
    return ensure_outfile_param(param="outfile", default=out_path)
