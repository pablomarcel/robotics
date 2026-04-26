# velocity_kinematics/tools/diagram.py
"""
Class diagram generator for the velocity_kinematics-kinematics module.

Features
--------
- OOP façade: DiagramTool + DiagramConfig
- Backends: DOT text, PlantUML text, optional Graphviz render (PNG/SVG/PDF)
- JSON model export
- Themed styles (light/dark), package clustering, legend, rankdir
- Back-compatible helpers `render_dot()` and `render_puml()` for quick use.

CLI Examples
------------
# DOT (file) and PUML (file)
python -m velocity_kinematics.tools.diagram dot --out velocity_kinematics/out/classes.dot
python -m velocity_kinematics.tools.diagram plantuml --out velocity_kinematics/out/classes.puml

# Graphviz (PNG) with high DPI (no size clamp)
python -m velocity_kinematics.tools.diagram graphviz --fmt png --dpi 300 --outstem classes

# All artifacts (tries graphviz; if missing it reports 'unavailable')
python -m velocity_kinematics.tools.diagram all
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, get_args, get_origin
import abc
import argparse
import importlib
import inspect
import json

__all__ = [
    "DiagramTool",
    "DiagramConfig",
    "render_dot",
    "render_puml",
]

# ---------------------------------------------------------------------------
# Defaults & Theme
# ---------------------------------------------------------------------------

_DEFAULT_PACKAGES: Tuple[str, ...] = (
    "velocity_kinematics.core",
    "velocity_kinematics.io",
    "velocity_kinematics.design",
    "velocity_kinematics.utils",
    "velocity_kinematics.app",
    "velocity_kinematics.apis",
    "velocity_kinematics.cli",
)

_THEMES = {
    "light": dict(
        graph_bg="white",
        cluster_color="#c7cedb",
        node_fill="#f7f9fc",
        node_border="#374151",
        iface_border="#6b7280",
        text="#111827",
        assoc="#374151",
        inherit="#111827",
    ),
    "dark": dict(
        graph_bg="#0f172a",
        cluster_color="#475569",
        node_fill="#111827",
        node_border="#cbd5e1",
        iface_border="#94a3b8",
        text="#e5e7eb",
        assoc="#cbd5e1",
        inherit="#e5e7eb",
    ),
}

# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def _sanitize_id(s: str) -> str:
    return s.replace(".", "_").replace(":", "_").replace("-", "_")


def _typename(tp) -> str:
    try:
        return tp.__name__
    except Exception:
        return str(tp).replace("typing.", "")


def _is_interface(cls: type) -> bool:
    try:
        return issubclass(cls, abc.ABC)
    except Exception:
        return False


def _class_members_summary(cls) -> Tuple[List[str], List[str]]:
    """Return (fields, methods) for a class (only methods declared on the class)."""
    ann = getattr(cls, "__annotations__", {}) or {}
    fields: List[str] = [f"{name}: {_typename(tp)}" for name, tp in ann.items()]

    methods: List[str] = []
    for name, obj in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith("__") and name.endswith("__"):
            continue
        if getattr(obj, "__qualname__", "").split(".")[0] == cls.__name__:
            methods.append(name)
    return fields, methods


def _iter_classes(packages: Sequence[str]):
    """Yield (module_name, class_obj) for classes defined in the given package prefixes."""
    prefixes = tuple(packages)
    seen: Set[type] = set()
    for mod_name in packages:
        mod = importlib.import_module(mod_name)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if not getattr(obj, "__module__", "").startswith(prefixes):
                continue
            if obj in seen:
                continue
            seen.add(obj)
            yield obj.__module__, obj

# --------------------------- association inference ---------------------------

_MULT_MANY = "0..*"
_MULT_OPT = "0..1"
_MULT_ONE = "1"

def _is_optional_of(tp, target: type) -> bool:
    origin = get_origin(tp)
    if origin is None:
        return False
    # Optional[T] is Union[T, NoneType]
    if origin is Optional:
        return target in get_args(tp)
    if origin is not None and "Union" in str(origin):
        return target in get_args(tp)
    return False

def _is_sequence_of(tp, target: type) -> bool:
    origin = get_origin(tp)
    if origin in (list, set):
        args = get_args(tp)
        return len(args) == 1 and args[0] is target
    if origin and any(s in str(origin) for s in ("Sequence", "List", "Set")):
        args = get_args(tp)
        return len(args) == 1 and args[0] is target
    if origin is tuple:
        args = get_args(tp)
        return any(a is target for a in args)
    return False

def _collect_associations(cls: type, in_scope: Dict[str, type]) -> List[Tuple[str, str, str, str]]:
    """
    Find associations from `cls` to other classes in `in_scope`.

    Returns list of (src_id, dst_id, label, multiplicity)
    """
    out: List[Tuple[str, str, str, str]] = []
    ann = getattr(cls, "__annotations__", {}) or {}
    if not ann:
        return out

    by_name: Dict[str, type] = {c.__name__: c for c in in_scope.values()}

    for attr, tp in ann.items():
        target = None
        mult = None

        if isinstance(tp, type) and tp in in_scope.values():
            target, mult = tp, _MULT_ONE
        elif isinstance(tp, str):
            t = by_name.get(tp)
            if t:
                target, mult = t, _MULT_ONE
        else:
            for cand in in_scope.values():
                if _is_sequence_of(tp, cand):
                    target, mult = cand, _MULT_MANY
                    break
                if _is_optional_of(tp, cand):
                    target, mult = cand, _MULT_OPT
                    break
            else:
                origin = get_origin(tp)
                args = get_args(tp)
                if origin is None and args:
                    for cand in in_scope.values():
                        if cand in args:
                            target, mult = cand, _MULT_ONE
                            break
                elif origin in (list, set) and args:
                    for cand in in_scope.values():
                        if args[0] is cand:
                            target, mult = cand, _MULT_MANY
                            break

        if target:
            src_id = _sanitize_id(f"{cls.__module__}.{cls.__name__}")
            dst_id = _sanitize_id(f"{target.__module__}.{target.__name__}")
            out.append((src_id, dst_id, attr, mult or _MULT_ONE))
    return out

# ---------------------------------------------------------------------------
# Minimal model (JSON-friendly)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassInfo:
    qualname: str
    modname: str
    name: str
    bases: Tuple[str, ...]
    fields: Tuple[str, ...]
    methods: Tuple[str, ...]
    is_interface: bool
    is_dataclass: bool


@dataclass(frozen=True)
class Model:
    package_roots: Tuple[str, ...]
    classes: Tuple[ClassInfo, ...]


# ---------------------------------------------------------------------------
# Rendering backends
# ---------------------------------------------------------------------------

def _build_model(packages: Sequence[str]) -> Model:
    classes: List[ClassInfo] = []
    for module_name, cls in _iter_classes(packages):
        fields, methods = _class_members_summary(cls)
        ci = ClassInfo(
            qualname=f"{cls.__module__}.{cls.__name__}",
            modname=module_name,
            name=cls.__name__,
            bases=tuple(f"{b.__module__}.{b.__name__}" for b in cls.__bases__ if hasattr(b, "__name__")),
            fields=tuple(fields),
            methods=tuple(methods),
            is_interface=_is_interface(cls),
            is_dataclass=is_dataclass(cls),
        )
        classes.append(ci)
    return Model(tuple(packages), tuple(classes))


def _render_dot(
    model: Model,
    *,
    theme: str = "light",
    rankdir: str = "LR",
    cluster_by_module: bool = True,
    add_legend: bool = False,
) -> str:
    colors = _THEMES.get(theme, _THEMES["light"])

    # group by module
    mod2classes: Dict[str, List[ClassInfo]] = {}
    for c in model.classes:
        mod2classes.setdefault(c.modname, []).append(c)

    # quick scope for associations
    qn_to_ci: Dict[str, ClassInfo] = {c.qualname: c for c in model.classes}

    lines: List[str] = []
    lines.append("digraph VelocityDiagram {")
    lines.append(f'  graph [rankdir={rankdir}, fontsize=10, bgcolor="{colors["graph_bg"]}"];')
    lines.append(
        '  node  [shape=record, fontname="Helvetica", fontsize=10, style="rounded,filled"];'
        f'  node  [color="{colors["node_border"]}", fillcolor="{colors["node_fill"]}", fontcolor="{colors["text"]}"];'
    )
    lines.append(
        '  edge  [arrowsize=0.9, fontname="Helvetica", fontsize=9];'
        f'  edge  [color="{colors["assoc"]}", fontcolor="{colors["text"]}"];'
    )

    # nodes
    def add_node(ci: ClassInfo):
        title = ci.name + (" «dataclass»" if ci.is_dataclass else "")
        if ci.is_interface:
            node_style = f'style="dashed,filled", color="{colors["iface_border"]}", fillcolor="{colors["node_fill"]}"'
        else:
            node_style = f'color="{colors["node_border"]}", fillcolor="{colors["node_fill"]}"'
        fields_block = "\\l".join(ci.fields) + ("\\l" if ci.fields else "")
        methods_block = "\\l".join(ci.methods) + ("\\l" if ci.methods else "")
        label = "{%s|%s|%s}" % (title, fields_block, methods_block)
        cid = _sanitize_id(ci.qualname)
        lines.append(f'    {cid} [label="{label}" {node_style}];')

    if cluster_by_module:
        for modname, items in sorted(mod2classes.items()):
            cluster_id = _sanitize_id(f"cluster_{modname}")
            lines.append(f'  subgraph {cluster_id} {{')
            lines.append(f'    label="{modname}";')
            lines.append(f'    style="rounded"; color="{colors["cluster_color"]}"; fontname="Helvetica"; fontsize=9;')
            for ci in items:
                add_node(ci)
            lines.append("  }")
    else:
        for ci in model.classes:
            add_node(ci)

    # inheritance / realization
    qn_set = set(qn_to_ci.keys())
    for ci in model.classes:
        child_id = _sanitize_id(ci.qualname)
        for bqn in ci.bases:
            if bqn in qn_set:
                base_id = _sanitize_id(bqn)
                base_iface = qn_to_ci[bqn].is_interface
                if base_iface:
                    lines.append(f'  {child_id} -> {base_id} [arrowhead=empty, style="dashed", color="{colors["inherit"]}"];')
                else:
                    lines.append(f'  {child_id} -> {base_id} [arrowhead=empty, color="{colors["inherit"]}"];')

    # associations (recompute via actual types; we need runtime classes)
    in_scope_runtime: Dict[str, type] = {}
    for _, cls in _iter_classes(model.package_roots):
        in_scope_runtime[f"{cls.__module__}.{cls.__name__}"] = cls

    for _, cls in _iter_classes(model.package_roots):
        assoc = _collect_associations(cls, in_scope_runtime)
        for src_id, dst_id, label, mult in assoc:
            lines.append(
                f'  {src_id} -> {dst_id} [arrowhead="vee", label="{label}", headlabel="{mult}", labelfontsize=9];'
            )

    if add_legend:
        lines.append('  subgraph cluster_legend {')
        lines.append('    label="Legend"; fontsize=9; style="rounded"; color="#9ca3af";')
        lines.append('    lg1 [shape=plain, label=<')
        lines.append('      <table border="0" cellborder="1" cellspacing="0" cellpadding="4">')
        lines.append('        <tr><td><b>Notation</b></td><td>Meaning</td></tr>')
        lines.append('        <tr><td>solid triangle</td><td>Inheritance</td></tr>')
        lines.append('        <tr><td>dashed triangle</td><td>Realization (implements ABC)</td></tr>')
        lines.append('        <tr><td>A → "mult" B : attr</td><td>Association & multiplicity</td></tr>')
        lines.append('      </table>')
        lines.append('    >];')
        lines.append('  }')

    lines.append("}")
    return "\n".join(lines)


def _render_puml(
    model: Model,
    *,
    theme: str = "light",
    cluster_by_module: bool = True,
    add_legend: bool = False,
) -> str:
    colors = _THEMES.get(theme, _THEMES["light"])

    mod2classes: Dict[str, List[ClassInfo]] = {}
    for c in model.classes:
        mod2classes.setdefault(c.modname, []).append(c)

    qn_to_ci: Dict[str, ClassInfo] = {c.qualname: c for c in model.classes}
    qn_set = set(qn_to_ci.keys())

    lines: List[str] = []
    lines.append("@startuml")
    lines.append("left to right direction")
    lines.append("skinparam classAttributeIconSize 0")
    lines.append(f'skinparam BackgroundColor {colors["graph_bg"]}')
    lines.append("skinparam packageStyle rectangle")
    lines.append(
        f'skinparam DefaultTextAlignment left\n'
        f'skinparam ClassFontColor {colors["text"]}\n'
        f'skinparam ClassBackgroundColor {colors["node_fill"]}\n'
        f'skinparam ClassBorderColor {colors["node_border"]}\n'
        f'skinparam ArrowColor {colors["assoc"]}'
    )

    def add_class(ci: ClassInfo):
        kind = "interface" if ci.is_interface else "class"
        title = ci.name + (" <<dataclass>>" if ci.is_dataclass else "")
        cid = _sanitize_id(ci.qualname)
        lines.append(f'{kind} "{title}" as {cid} {{')
        for f in ci.fields:
            lines.append(f"  {f}")
        if ci.methods:
            if ci.fields:
                lines.append("--")
            for m in ci.methods:
                lines.append(f"  {m}()")
        lines.append("}")

    if cluster_by_module:
        for modname, items in sorted(mod2classes.items()):
            lines.append(f'package "{modname}" {{')
            for ci in items:
                add_class(ci)
            lines.append("}")
    else:
        for ci in model.classes:
            add_class(ci)

    # inheritance / realization
    for ci in model.classes:
        child_id = _sanitize_id(ci.qualname)
        for bqn in ci.bases:
            if bqn in qn_set:
                base_id = _sanitize_id(bqn)
                if qn_to_ci[bqn].is_interface:
                    lines.append(f"{child_id} ..|> {base_id}")
                else:
                    lines.append(f"{child_id} --|> {base_id}")

    # associations via runtime pass
    in_scope_runtime: Dict[str, type] = {}
    for _, cls in _iter_classes(model.package_roots):
        in_scope_runtime[f"{cls.__module__}.{cls.__name__}"] = cls
    for _, cls in _iter_classes(model.package_roots):
        assoc = _collect_associations(cls, in_scope_runtime)
        for src_id, dst_id, label, mult in assoc:
            lines.append(f'{src_id} --> "{mult}" {dst_id} : {label}')

    if add_legend:
        lines.append("legend left")
        lines.append("| Notation | Meaning |")
        lines.append("| solid ─┤▷ | Inheritance |")
        lines.append("| dashed ─┄┤▷ | Realization (ABC) |")
        lines.append("| A --> \"mult\" B : attr | Association & multiplicity |")
        lines.append("endlegend")

    lines.append("@enduml")
    return "\n".join(lines)


def _graphviz_render(
    dot_src: str,
    *,
    out_dir: Path,
    out_stem: str,
    fmt: str = "png",
    dpi: int = 200,
    engine: str = "dot",
) -> str:
    """
    Render DOT using python-graphviz (no explicit size, so no PNG size clamp).
    """
    try:
        from graphviz import Source  # type: ignore
    except Exception as e:
        raise RuntimeError("python 'graphviz' package is not installed") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    src = Source(dot_src, engine=engine, format=fmt)
    # remove any size constraints; allow high dpi
    src.graph_attr.update({
        "overlap": "false",
        "splines": "true",
        "bgcolor": "transparent",
    })
    path = src.render(filename=str((out_dir / out_stem).as_posix()), cleanup=True, format=fmt, quiet=True)
    return path

# ---------------------------------------------------------------------------
# Public Facade
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagramConfig:
    packages: Tuple[str, ...] = _DEFAULT_PACKAGES
    out_dir: Path = Path("velocity_kinematics/out")
    theme: str = "light"
    rankdir: str = "LR"
    cluster_by_module: bool = True
    add_legend: bool = False


class DiagramTool:
    """
    Facade service for class diagrams of the `velocity_kinematics` package.
    """

    def __init__(self, config: Optional[DiagramConfig] = None) -> None:
        self.cfg = config or DiagramConfig()
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Discovery
    def discover(self) -> Model:
        return _build_model(self.cfg.packages)

    # Emitters
    def emit_dot(self, *, out_file: Optional[str] = None) -> str:
        model = self.discover()
        dot = _render_dot(
            model,
            theme=self.cfg.theme,
            rankdir=self.cfg.rankdir,
            cluster_by_module=self.cfg.cluster_by_module,
            add_legend=self.cfg.add_legend,
        )
        if out_file:
            path = (self.cfg.out_dir / out_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dot, encoding="utf-8")
            return str(path)
        return dot

    def emit_plantuml(self, *, out_file: Optional[str] = None) -> str:
        model = self.discover()
        puml = _render_puml(
            model,
            theme=self.cfg.theme,
            cluster_by_module=self.cfg.cluster_by_module,
            add_legend=self.cfg.add_legend,
        )
        if out_file:
            path = (self.cfg.out_dir / out_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(puml, encoding="utf-8")
            return str(path)
        return puml

    def export_model_json(self, *, out_file: str = "classes.json") -> str:
        model = self.discover()
        path = self.cfg.out_dir / out_file
        payload = dict(
            package_roots=list(model.package_roots),
            classes=[asdict(c) for c in model.classes],
        )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    # Optional: render using python-graphviz (no binary needed)
    def render_graphviz(self, *, fmt: str = "png", dpi: int = 200,
                        out_stem: str = "classes", engine: str = "dot") -> str:
        dot = self.emit_dot()  # as text
        return _graphviz_render(
            dot, out_dir=self.cfg.out_dir, out_stem=out_stem, fmt=fmt, dpi=dpi, engine=engine
        )

    def render_all(self) -> Dict[str, str]:
        """
        Produce a set of outputs. Graphviz is best-effort.
        """
        artifacts: Dict[str, str] = {}
        artifacts["json"] = self.export_model_json()
        artifacts["dot"] = self.emit_dot(out_file="classes.dot")
        artifacts["plantuml"] = self.emit_plantuml(out_file="classes.puml")
        try:
            artifacts["graphviz"] = self.render_graphviz(fmt="png", dpi=260, out_stem="classes")
        except RuntimeError:
            artifacts["graphviz"] = "unavailable (python-graphviz not installed)"
        return artifacts

# ---------------------------------------------------------------------------
# Back-compat convenience wrappers
# ---------------------------------------------------------------------------

def render_dot(
    packages: Sequence[str] = _DEFAULT_PACKAGES,
    *,
    theme: str = "light",
    rankdir: str = "LR",
    cluster_by_module: bool = True,
    add_legend: bool = False,
) -> str:
    """
    Emit Graphviz DOT text for the given packages.
    """
    model = _build_model(packages)
    return _render_dot(
        model,
        theme=theme,
        rankdir=rankdir,
        cluster_by_module=cluster_by_module,
        add_legend=add_legend,
    )


def render_puml(
    packages: Sequence[str] = _DEFAULT_PACKAGES,
    *,
    theme: str = "light",
    cluster_by_module: bool = True,
    add_legend: bool = False,
) -> str:
    """
    Emit PlantUML text for the given packages.
    """
    model = _build_model(packages)
    return _render_puml(
        model,
        theme=theme,
        cluster_by_module=cluster_by_module,
        add_legend=add_legend,
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="velocity_kinematics.tools.diagram", description="Class diagram generator (velocity_kinematics module)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(a):
        a.add_argument("--packages", default=",".join(_DEFAULT_PACKAGES),
                       help="Comma-separated package prefixes to include.")
        a.add_argument("--outdir", default="velocity_kinematics/out", help="Output directory (default: velocity_kinematics/out)")
        a.add_argument("--theme", default="light", choices=list(_THEMES.keys()))
        a.add_argument("--rankdir", default="LR", choices=["LR", "TB"])
        a.add_argument("--legend", action="store_true", help="Add a legend box")
        a.add_argument("--no-cluster", action="store_true", help="Do not group classes by module")

    # dot
    d = sub.add_parser("dot", help="Emit Graphviz DOT text")
    add_common(d)
    d.add_argument("--out", default="classes.dot", help="Output filename (within outdir). If blank, prints to stdout.")

    # plantuml
    u = sub.add_parser("plantuml", help="Emit PlantUML text")
    add_common(u)
    u.add_argument("--out", default="classes.puml", help="Output filename (within outdir). If blank, prints to stdout.")

    # json
    j = sub.add_parser("json", help="Export discovered model as JSON")
    add_common(j)
    j.add_argument("--out", default="classes.json", help="Output filename (within outdir)")

    # graphviz
    g = sub.add_parser("graphviz", help="Render Graphviz image via python-graphviz (no size clamp)")
    add_common(g)
    g.add_argument("--fmt", default="png", choices=["png", "svg", "pdf"])
    g.add_argument("--dpi", default=220, type=int, help="Raster DPI (PNG); ignored for SVG/PDF")
    g.add_argument("--outstem", default="classes", help="Output stem (extension added by renderer)")

    # all
    a = sub.add_parser("all", help="Emit JSON, DOT, PlantUML and try Graphviz")
    add_common(a)

    return p


def _parse_packages(s: str) -> Tuple[str, ...]:
    return tuple([x.strip() for x in s.split(",") if x.strip()] or _DEFAULT_PACKAGES)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)

    cfg = DiagramConfig(
        packages=_parse_packages(args.packages),
        out_dir=Path(args.outdir),
        theme=args.theme,
        rankdir=args.rankdir,
        cluster_by_module=not args.no_cluster,
        add_legend=args.legend,
    )
    tool = DiagramTool(cfg)

    if args.cmd == "dot":
        if args.out:
            print(tool.emit_dot(out_file=args.out))
        else:
            print(tool.emit_dot())
        return

    if args.cmd == "plantuml":
        if args.out:
            print(tool.emit_plantuml(out_file=args.out))
        else:
            print(tool.emit_plantuml())
        return

    if args.cmd == "json":
        print(tool.export_model_json(out_file=args.out))
        return

    if args.cmd == "graphviz":
        try:
            print(tool.render_graphviz(fmt=args.fmt, dpi=args.dpi, out_stem=args.outstem))
        except RuntimeError as e:
            raise SystemExit(str(e))
        return

    if args.cmd == "all":
        arts = tool.render_all()
        for k, v in arts.items():
            print(f"{k}: {v}")
        return


if __name__ == "__main__":  # pragma: no cover
    main()
