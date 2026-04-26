# time/tools/diagram.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable

PUML_HEADER = "@startuml time\n!pragma useIntermediatePackages false\n"
PUML_FOOTER = "@enduml\n"

def emit_puml(classes: Iterable[str], edges: Iterable[tuple[str, str]]) -> str:
    s = [PUML_HEADER]
    for c in classes:
        s += [f"class {c} {{}}\n"]
    for a, b in edges:
        s += [f"{a} --> {b}\n"]
    s += [PUML_FOOTER]
    return "".join(s)

def write_puml(path: str | Path, content: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
