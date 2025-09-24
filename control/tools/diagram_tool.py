from __future__ import annotations
import inspect, pkgutil
from pathlib import Path
import control as thispkg  # our package name is 'control'
from .trace import track
from typing import List

@track("diagram.emit")
def emit_mermaid(out_name="classes"):
    """Zero-dep Mermaid class diagram (best-effort)."""
    from .. import core, design, apis, app
    modules = [core, design, apis, app]
    lines: List[str] = ["```mermaid","classDiagram"]
    for m in modules:
        for name, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__.startswith("control."):
                lines.append(f"class {obj.__module__.split('.')[-1]}_{name}")
    lines.append("```")
    OUT = Path(__file__).resolve().parents[1] / "out"
    OUT.mkdir(exist_ok=True, parents=True)
    p = OUT / f"{out_name}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
