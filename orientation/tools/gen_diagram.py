# orientation/tools/gen_diagram.py
"""
Generate class diagrams (PlantUML / DOT / Mermaid) by introspecting the
orientation.* package. We infer relationships from:
- inheritance (subclass -> baseclass)
- composition ("has"): class attributes/fields typed as another in-project class
- usage ("uses"): method parameter/return type annotations referencing another in-project class

Examples
--------
# Default: PlantUML to orientation/out/class_diagram.puml
python -m orientation.tools.gen_diagram

# Mermaid instead, with fewer modules and no members (cleaner map)
python -m orientation.tools.gen_diagram --format mmd \
  --modules orientation.core orientation.utils orientation.io \
  --no-members

# Graphviz DOT and show private members
python -m orientation.tools.gen_diagram --format dot --show-private
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import re
import sys
import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, get_type_hints

# fallbacks if run outside package context
try:
    from ..io import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = Path(__file__).resolve().parents[2] / "orientation" / "out"


# ---------------------- model ----------------------

@dataclass
class MethodInfo:
    name: str
    signature: str
    annotations: Dict[str, str]  # param/return -> type str
    is_private: bool

@dataclass
class ClassInfo:
    qname: str
    module: str
    name: str
    bases: List[str]                 # base simple names or qualified
    attrs: Dict[str, str]            # attr -> type str
    methods: List[MethodInfo] = field(default_factory=list)

@dataclass
class Relationship:
    src: str          # class qname
    dst: str          # class qname
    kind: str         # 'inherits' | 'has' | 'uses'


# ---------------------- config ----------------------

@dataclass
class Config:
    modules: Sequence[str]
    fmt: str = "puml"                # 'puml' | 'dot' | 'mmd'
    out_dir: Path = OUT_DIR
    filename: Optional[str] = None   # auto from fmt if None
    include_private: bool = False
    include_members: bool = True     # show attrs+methods blocks
    project_prefix: str = "orientation."

    def output_path(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = self.filename
        if not name:
            ext = {"puml": "puml", "dot": "dot", "mmd": "mmd"}[self.fmt]
            name = f"class_diagram.{ext}"
        return self.out_dir / name

    def is_project_mod(self, modname: str) -> bool:
        return modname.startswith(self.project_prefix)


# ---------------------- introspection ----------------------

class Introspector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._visited: set[str] = set()

    def discover(self) -> Dict[str, ClassInfo]:
        classes: Dict[str, ClassInfo] = {}
        for root in self.cfg.modules:
            try:
                mod = importlib.import_module(root)
            except Exception:
                continue
            self._walk(mod, classes)
        return classes

    def _walk(self, mod: types.ModuleType, classes: Dict[str, ClassInfo]):
        if mod.__name__ in self._visited:
            return
        self._visited.add(mod.__name__)

        # classes defined in this module
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ != mod.__name__:
                continue
            ci = self._class_info(obj)
            classes[ci.qname] = ci

        # recurse into submodules if package
        if hasattr(mod, "__path__"):
            for sub in pkgutil.iter_modules(mod.__path__, mod.__name__ + "."):
                try:
                    submod = importlib.import_module(sub.name)
                except Exception:
                    continue
                self._walk(submod, classes)

    def _class_info(self, cls: type) -> ClassInfo:
        module = cls.__module__
        name = cls.__name__
        qname = f"{module}.{name}"
        bases = [b.__name__ for b in cls.__bases__ if hasattr(b, "__name__")]

        # attributes: dataclass fields + __annotations__
        attrs: Dict[str, str] = {}
        if is_dataclass(cls):
            try:
                for f in fields(cls):
                    attrs[f.name] = friendly_type(f.type)
            except Exception:
                pass
        try:
            ann = get_type_hints(cls, include_extras=False)
            for a, t in ann.items():
                attrs.setdefault(a, friendly_type(t))
        except Exception:
            pass

        # methods
        methods: List[MethodInfo] = []
        for mname, member in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not self.cfg.include_private and mname.startswith("_"):
                continue
            try:
                sig = str(inspect.signature(member))
            except Exception:
                sig = "(...)"
            try:
                hints = get_type_hints(member, include_extras=False)
                hints = {k: friendly_type(v) for k, v in hints.items()}
            except Exception:
                hints = {}
            methods.append(MethodInfo(
                name=mname,
                signature=sig,
                annotations=hints,
                is_private=mname.startswith("_"),
            ))

        return ClassInfo(qname=qname, module=module, name=name, bases=bases, attrs=attrs, methods=methods)


# ---------------------- inference ----------------------

class RelationshipInferer:
    def __init__(self, cfg: Config, classes: Dict[str, ClassInfo]):
        self.cfg = cfg
        self.classes = classes
        # short-name -> fully qualified (first hit wins)
        self.short2full: Dict[str, str] = {}
        for q, ci in classes.items():
            self.short2full.setdefault(ci.name, q)

    def infer(self) -> List[Relationship]:
        rels: dict[tuple[str, str, str], Relationship] = {}

        def add(src: str, dst: Optional[str], kind: str):
            if not dst:
                return
            if not (src in self.classes and dst in self.classes):
                return
            rels[(src, dst, kind)] = Relationship(src, dst, kind)

        # inheritance
        for q, ci in self.classes.items():
            for b in ci.bases:
                add(q, self._resolve(b), "inherits")

        # composition from attributes
        for q, ci in self.classes.items():
            for _, tname in ci.attrs.items():
                add(q, self._resolve_type_tokens(tname), "has")

        # usage from method annotations
        for q, ci in self.classes.items():
            for mi in ci.methods:
                for _, tname in mi.annotations.items():
                    add(q, self._resolve_type_tokens(tname), "uses")

        return list(rels.values())

    def _resolve(self, name: str) -> Optional[str]:
        # already qualified?
        if name in self.classes:
            return name
        # simple name?
        return self.short2full.get(name)

    def _resolve_type_tokens(self, type_str: str) -> Optional[str]:
        # strip typing extras like List[SO3], Optional[Quaternion], dict[str, AxisAngle]
        # tokenize by non-word chars, then try each token
        for tok in re.split(r"[^A-Za-z0-9_.]+", type_str):
            if not tok:
                continue
            if tok in self.classes:
                return tok
            if tok in self.short2full:
                return self.short2full[tok]
        return None


def friendly_type(t) -> str:
    try:
        if hasattr(t, "__name__"):
            return t.__name__
        s = str(t)
        return s.replace("typing.", "")
    except Exception:
        return "Any"


# ---------------------- renderers ----------------------

class RenderPUML:
    def __init__(self, include_members: bool):
        self.include_members = include_members

    def render(self, classes: Dict[str, ClassInfo], rels: List[Relationship]) -> str:
        out: List[str] = ["@startuml", "skinparam classAttributeIconSize 0", "hide empty members"]
        for ci in classes.values():
            if self.include_members:
                out.append(f'class "{ci.qname}" as {self._alias(ci.qname)} {{')
                for a, t in sorted(ci.attrs.items()):
                    out.append(f"  + {a}: {t}")
                for m in sorted(ci.methods, key=lambda m: m.name):
                    out.append(f"  + {m.name}{m.signature}")
                out.append("}")
            else:
                out.append(f'class "{ci.qname}" as {self._alias(ci.qname)}')
        for r in rels:
            s = self._alias(r.src); d = self._alias(r.dst)
            if r.kind == "inherits":
                out.append(f"{s} <|-- {d}")
            elif r.kind == "has":
                out.append(f"{s} *-- {d}")
            else:
                out.append(f"{s} ..> {d}")
        out.append("@enduml")
        return "\n".join(out)

    @staticmethod
    def _alias(qname: str) -> str:
        return qname.replace(".", "_")


class RenderDOT:
    def __init__(self, include_members: bool):
        self.include_members = include_members

    def render(self, classes: Dict[str, ClassInfo], rels: List[Relationship]) -> str:
        out: List[str] = ['digraph G {', 'rankdir=LR;', 'node [shape=record, fontsize=10];']
        for ci in classes.values():
            if self.include_members:
                attrs = r"\l".join(f"+ {k}: {v}" for k, v in sorted(ci.attrs.items()))
                meths = r"\l".join(f"+ {m.name}{m.signature}" for m in sorted(ci.methods, key=lambda m: m.name))
                label = f"{{{ci.qname}|{attrs}|{meths}}}"
            else:
                label = f"{{{ci.qname}}}"
            out.append(f'"{ci.qname}" [label="{label}"];')
        for r in rels:
            style = {"inherits": "arrowhead=onormal", "has": "arrowhead=diamond", "uses": "style=dashed,arrowhead=vee"}[r.kind]
            out.append(f'"{r.src}" -> "{r.dst}" [{style}];')
        out.append("}")
        return "\n".join(out)


class RenderMMD:
    def __init__(self, include_members: bool):
        self.include_members = include_members

    def render(self, classes: Dict[str, ClassInfo], rels: List[Relationship]) -> str:
        out: List[str] = ["classDiagram"]
        for ci in classes.values():
            if self.include_members:
                out.append(f'class "{ci.qname}" {{')
                for a, t in sorted(ci.attrs.items()):
                    out.append(f"  + {a}: {t}")
                for m in sorted(ci.methods, key=lambda m: m.name):
                    out.append(f"  + {m.name}{m.signature}")
                out.append("}")
            else:
                out.append(f'class "{ci.qname}"')
        for r in rels:
            s, d = r.src, r.dst
            if r.kind == "inherits":
                out.append(f'"{s}" <|-- "{d}"')
            elif r.kind == "has":
                out.append(f'"{s}" *-- "{d}"')
            else:
                out.append(f'"{s}" ..> "{d}"')
        return "\n".join(out)


# ---------------------- CLI app ----------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate class diagrams for orientation.*")
    p.add_argument("--modules", nargs="*", default=[
        "orientation.core", "orientation.utils", "orientation.io",
        "orientation.apis", "orientation.cli", "orientation.design", "orientation.app"
    ], help="Root modules/packages to scan.")
    p.add_argument("--format", dest="fmt", choices=["puml", "dot", "mmd"], default="puml")
    p.add_argument("--out", type=str, help="Output directory (default orientation/out).")
    p.add_argument("--name", type=str, help="Output filename (default based on format).")
    p.add_argument("--show-private", action="store_true", help="Include private methods (prefixed with _).")
    p.add_argument("--no-members", action="store_true", help="Hide attributes and methods blocks, show only relations.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config(
        modules=args.modules,
        fmt=args.fmt,
        out_dir=Path(args.out) if args.out else OUT_DIR,
        filename=args.name,
        include_private=bool(args.show_private),
        include_members=not bool(args.no_members),
    )

    classes = Introspector(cfg).discover()
    rels = RelationshipInferer(cfg, classes).infer()

    if cfg.fmt == "puml":
        text = RenderPUML(cfg.include_members).render(classes, rels)
    elif cfg.fmt == "dot":
        text = RenderDOT(cfg.include_members).render(classes, rels)
    else:
        text = RenderMMD(cfg.include_members).render(classes, rels)

    out_path = cfg.output_path()
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {cfg.fmt.upper()} -> {out_path}")


if __name__ == "__main__":
    main()
