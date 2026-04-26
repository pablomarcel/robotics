"""
orientation_kinematics.design (OO version)
-------------------------------
Object-oriented class diagram generator for the orientation_kinematics package.

Design
------
- DiagramConfig: configuration (targets, output directory, filters)
- ClassIntrospector: discover classes & relationships from modules
- DiagramBuilder (abstract): pluggable emitters
- MermaidBuilder / GraphvizBuilder: concrete emitters
- DiagramGenerator: orchestrates introspection + emit to files

Outputs
-------
- Graphviz DOT  : <out_dir>/class_diagram.dot
- Mermaid (mmd) : <out_dir>/class_diagram.mmd
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, get_args, get_origin
import inspect
import types


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class DiagramConfig:
    """Configuration for class diagram generation."""
    out_dir: Path
    modules: Sequence[types.ModuleType]
    # Optional: include only classes from these module-name prefixes
    include_prefixes: Sequence[str] = field(default_factory=lambda: ())
    # Optional: exclude class names
    exclude_class_names: Sequence[str] = field(default_factory=lambda: ())

    def should_include_module(self, module_name: str) -> bool:
        if not self.include_prefixes:
            return True
        return any(module_name.startswith(p) for p in self.include_prefixes)

    def should_include_class(self, cls: type) -> bool:
        if self.exclude_class_names and cls.__name__ in self.exclude_class_names:
            return False
        return self.should_include_module(cls.__module__)


# --------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------

@dataclass
class ClassInfo:
    name: str
    module: str
    annotations: Dict[str, str]  # field -> type string


@dataclass
class Relationship:
    a: str   # class name
    b: str   # class name
    kind: str = "association"  # simple association


class ClassIntrospector:
    """Find classes and simple relationships using type annotations."""

    def __init__(self, config: DiagramConfig):
        self.config = config

    def collect(self) -> Tuple[List[ClassInfo], List[Relationship]]:
        classes = self._discover_classes()
        relations = self._discover_relationships(classes)
        return classes, relations

    def _discover_classes(self) -> List[ClassInfo]:
        out: List[ClassInfo] = []
        for mod in self.config.modules:
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if not self.config.should_include_class(obj):
                    continue
                if obj.__module__.startswith(mod.__name__):
                    ann = self._simplify_annotations(getattr(obj, "__annotations__", {}))
                    out.append(ClassInfo(name=obj.__name__, module=obj.__module__, annotations=ann))
        # de-dup by name (prefer first occurrence)
        seen = set()
        dedup: List[ClassInfo] = []
        for c in out:
            if c.name in seen:
                continue
            seen.add(c.name)
            dedup.append(c)
        return dedup

    def _simplify_annotations(self, annotations: Dict[str, object]) -> Dict[str, str]:
        def as_str(tp: object) -> str:
            if isinstance(tp, type):
                return tp.__name__
            origin = get_origin(tp)
            if origin is None:
                return str(tp)
            args = ", ".join(as_str(a) for a in get_args(tp))
            return f"{getattr(origin, '__name__', str(origin))}[{args}]"
        return {k: as_str(v) for k, v in annotations.items()}

    def _discover_relationships(self, classes: List[ClassInfo]) -> List[Relationship]:
        names = {c.name for c in classes}
        rels: List[Relationship] = []
        for c in classes:
            for _, tname in c.annotations.items():
                # crude: if the string includes another class name, relate them
                for other in names:
                    if other == c.name:
                        continue
                    # Avoid substring traps by checking word boundaries-ish
                    if tname == other or tname.startswith(other + "[") or f" {other}" in tname or f",{other}" in tname:
                        rels.append(Relationship(a=c.name, b=other, kind="association"))
        # de-dup pairs
        uniq = {(r.a, r.b, r.kind) for r in rels}
        return [Relationship(*t) for t in uniq]


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

class DiagramBuilder:
    """Abstract diagram emitter."""
    def build(self, classes: List[ClassInfo], rels: List[Relationship]) -> str:
        raise NotImplementedError


class GraphvizBuilder(DiagramBuilder):
    """Render to Graphviz DOT with record-shaped nodes."""
    def build(self, classes: List[ClassInfo], rels: List[Relationship]) -> str:
        lines = ["digraph G {", "  rankdir=LR;", '  node [shape=record, fontsize=10];']
        for c in classes:
            fields = "\\l".join(f"{name}: {typ}" for name, typ in c.annotations.items()) + ("\\l" if c.annotations else "")
            label = f"{{{c.name}|{fields}}}"
            lines.append(f'  {c.name} [label="{label}"];')
        for r in rels:
            lines.append(f"  {r.a} -> {r.b} [arrowhead=vee];")
        lines.append("}")
        return "\n".join(lines)


class MermaidBuilder(DiagramBuilder):
    """Render to Mermaid classDiagram syntax."""
    def build(self, classes: List[ClassInfo], rels: List[Relationship]) -> str:
        lines = ["classDiagram"]
        for c in classes:
            lines.append(f"class {c.name} {{")
            if c.annotations:
                for name, typ in c.annotations.items():
                    lines.append(f"  +{name} {typ}")
            lines.append("}")
        for r in rels:
            # simple association
            lines.append(f"{r.a} --> {r.b}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Generator (orchestrator)
# --------------------------------------------------------------------------

class DiagramGenerator:
    """High-level generator to emit multiple diagram formats to disk."""

    def __init__(self, config: DiagramConfig, *, builders: Optional[List[Tuple[str, DiagramBuilder]]] = None):
        self.config = config
        self.builders = builders or [
            ("class_diagram.dot", GraphvizBuilder()),
            ("class_diagram.mmd", MermaidBuilder()),
        ]
        self.introspector = ClassIntrospector(config)

    def generate(self) -> List[Path]:
        classes, rels = self.introspector.collect()
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for filename, builder in self.builders:
            text = builder.build(classes, rels)
            path = self.config.out_dir / filename
            path.write_text(text, encoding="utf-8")
            paths.append(path)
        return paths


# --------------------------------------------------------------------------
# Convenience shim used by CLI
# --------------------------------------------------------------------------

def generate_diagram(out_dir: Path) -> Tuple[Path, Path]:
    """Generate both DOT and Mermaid diagrams for orientation_kinematics.core."""
    # Local import to avoid cycles if design is imported early
    from . import core
    cfg = DiagramConfig(
        out_dir=Path(out_dir),
        modules=(core,),
        include_prefixes=(core.__name__,),
        exclude_class_names=(),
    )
    gen = DiagramGenerator(cfg)
    paths = gen.generate()
    # Return (dot, mmd) in stable order
    dot = next(p for p in paths if p.suffix == ".dot")
    mmd = next(p for p in paths if p.suffix == ".mmd")
    return dot, mmd
