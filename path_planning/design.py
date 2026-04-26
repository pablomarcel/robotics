from __future__ import annotations
"""
Design model + exporters for diagrams (class/dep graphs).

This module is the *programmatic* counterpart to tools/diagram.py.
It can:
  1) Introspect the `path_planning` package and build a simple design model.
  2) Export PlantUML and Mermaid class diagrams (pure Python, no deps).
  3) If available, call external tools (pyreverse/graphviz, plantuml).

Typical use:
    from path_planning.design import DesignManager
    dm = DesignManager()
    model = dm.build_model()                     # in-memory model
    puml = dm.export_plantuml(model, "path_planning/out/path_classes.puml")
    svg  = dm.render_with_plantuml(puml)         # optional, if plantuml is installed
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import inspect
import importlib
import pkgutil
import shutil
import subprocess
import sys

# --------- decorator to skip classes in diagrams

_EXCLUDE: Set[str] = set()

def design_exclude(cls):
    """Decorator to exclude a class from diagrams."""
    _EXCLUDE.add(f"{cls.__module__}.{cls.__name__}")
    return cls

# --------- model

@dataclass
class ClassInfo:
    qualname: str
    name: str
    module: str
    bases: List[str] = field(default_factory=list)
    attrs: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)

@dataclass
class DesignModel:
    classes: Dict[str, ClassInfo] = field(default_factory=dict)  # key = qualname
    edges: List[Tuple[str, str]] = field(default_factory=list)   # (child -> base)

# --------- manager

@dataclass
class DesignManager:
    root_pkg: str = "path_planning"
    out_dir: Path = Path(__file__).resolve().parent / "out"

    def _iter_modules(self):
        pkg = importlib.import_module(self.root_pkg)
        yield pkg
        pkg_path = Path(pkg.__file__).resolve().parent
        for mod in pkgutil.walk_packages([str(pkg_path)], prefix=self.root_pkg + "."):
            try:
                yield importlib.import_module(mod.name)
            except Exception:
                # best-effort; keep going
                continue

    def build_model(self) -> DesignModel:
        """Introspect package and return a design model."""
        model = DesignModel()
        for mod in self._iter_modules():
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                qn = f"{obj.__module__}.{obj.__name__}"
                if not qn.startswith(self.root_pkg + "."):  # ignore external classes
                    continue
                if qn in _EXCLUDE:
                    continue
                bases = [f"{b.__module__}.{b.__name__}" for b in obj.__bases__
                         if b is not object]
                attrs = [n for n,v in inspect.getmembers(obj)
                         if not n.startswith("_") and not inspect.isroutine(v)]
                methods = [n for n,v in inspect.getmembers(obj, inspect.isfunction)
                           if not n.startswith("_")]
                ci = ClassInfo(qn, obj.__name__, obj.__module__, bases, attrs, methods)
                model.classes[qn] = ci
                for b in bases:
                    model.edges.append((qn, b))
        return model

    # ---------- textual exports (no external binaries required)

    def export_plantuml(self, model: DesignModel, outfile: str | Path) -> Path:
        """Write a PlantUML class diagram file (.puml) and return its path_planning."""
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        def short(qn: str) -> str:
            return qn.split(".")[-1]

        lines = ["@startuml", "skinparam classAttributeIconSize 0"]
        # classes
        for ci in model.classes.values():
            lines.append(f"class {ci.name} {{")
            for a in ci.attrs[:12]:
                lines.append(f"  +{a}")
            for m in ci.methods[:12]:
                lines.append(f"  +{m}()")
            lines.append("}")
            # package note
            lines.append(f'package "{ci.module}" {{}}')
            lines.append(f"{ci.module} .. {ci.name}")
        # inheritance
        for child, base in model.edges:
            if base in model.classes:
                lines.append(f"{short(child)} --|> {short(base)}")
        lines.append("@enduml")
        outfile.write_text("\n".join(lines))
        return outfile

    def export_mermaid(self, model: DesignModel, outfile: str | Path) -> Path:
        """Write a Mermaid class diagram (.mmd)."""
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        def short(qn: str) -> str: return qn.split(".")[-1]
        lines = ["classDiagram"]
        for ci in model.classes.values():
            lines.append(f"class {ci.name} {{")
            for a in ci.attrs[:12]: lines.append(f"  +{a}")
            for m in ci.methods[:12]: lines.append(f"  +{m}()")
            lines.append("}")
        for child, base in model.edges:
            if base in model.classes:
                lines.append(f"{short(base)} <|-- {short(child)}")
        outfile.write_text("\n".join(lines))
        return outfile

    # ---------- external tool adapters (optional)

    def render_with_plantuml(self, puml_file: str | Path, fmt: str = "svg") -> Optional[Path]:
        """
        If `plantuml` is available in PATH, render a .puml to an image.
        Returns output path_planning or None if not available.
        """
        if shutil.which("plantuml") is None:
            return None
        puml = Path(puml_file)
        out = puml.with_suffix("." + fmt)
        subprocess.check_call(["plantuml", "-t" + fmt, str(puml)])
        return out if out.exists() else None

    def pyreverse_graphviz(self, fmt: str = "svg", out_name: str = "path_uml") -> Optional[Path]:
        """
        Drive pyreverse→graphviz like tools/diagram.py but through code.
        Returns output path_planning or None if not available.
        """
        if shutil.which("pyreverse") is None or shutil.which("dot") is None:
            return None
        # run in current working dir to emit classes_*.dot
        subprocess.check_call(["pyreverse", "-o", "dot", "-p", "path_pkg", self.root_pkg])
        dot = Path("classes_path_pkg.dot")
        if dot.exists():
            out = self.out_dir / (out_name + "." + fmt)
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(["dot", "-T"+fmt, "-o", str(out), str(dot)])
            return out
        return None
