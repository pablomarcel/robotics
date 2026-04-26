# motion_kinematics/design.py
"""
Class diagram utilities for the `motion_kinematics` package.

Goals
-----
- Keep it dependency-light and testable.
- Offer multiple output backends:
  • Graphviz (PNG/SVG)  – optional; pretty diagrams.
  • pyreverse wrapper    – optional; leverages pylint's tool if installed.
  • PlantUML / Mermaid   – text emitters that work great in docs.
  • JSON model           – for golden-file unit tests.

Typical usage
-------------
from motion_kinematics import design

# Discover classes (pure python)
model = design.discover(package="motion_kinematics")

# Render with Graphviz (if graphviz is installed)
out_png = design.render_graphviz(model, out_path="motion_kinematics/out/classes", fmt="png")

# Or use pyreverse, if pylint is available
design.generate_pyreverse(package="motion_kinematics", out_dir="motion_kinematics/out", fmt="png")

# Emit textual diagrams
design.emit_plantuml(model, "motion_kinematics/out/classes.puml")
design.emit_mermaid(model, "motion_kinematics/out/classes.mmd")

# Save JSON model for testing
design.save_model_json(model, "motion_kinematics/out/classes.json")

Testing tips
-----------
- The discovery function is deterministic; you can snapshot its JSON in pytest.
- Graphviz/pyreverse calls are optional; unit tests can skip if missing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Set, Tuple
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys

# ------------------------------- Data model ----------------------------------

@dataclass(frozen=True)
class ClassInfo:
    qualname: str     # "motion_kinematics.core.SE3"
    modname: str      # "motion_kinematics.core"
    name: str         # "SE3"
    bases: Tuple[str, ...]  # fully-qualified base names under same package (filtered)
    doc: str = ""


@dataclass(frozen=True)
class Model:
    package: str
    classes: Tuple[ClassInfo, ...]


# ------------------------------ Discovery API --------------------------------

def discover(package: str = "motion_kinematics") -> Model:
    """
    Discover classes under a given top-level package (e.g., 'motion_kinematics').

    Returns
    -------
    Model
        A deterministic model of classes and their intra-package inheritance.
    """
    classes: List[ClassInfo] = []
    pkg = importlib.import_module(package)

    # Walk submodules
    if hasattr(pkg, "__path__"):  # it's a package
        import pkgutil
        for _, modname, _ in pkgutil.walk_packages(pkg.__path__, package + "."):
            _collect_module_classes(modname, package, classes)
    else:
        _collect_module_classes(package, package, classes)

    # Sort deterministically
    classes.sort(key=lambda c: c.qualname)
    return Model(package=package, classes=tuple(classes))


def _collect_module_classes(modname: str, root_pkg: str, out: List[ClassInfo]) -> None:
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return  # best-effort: skip import failures

    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if not obj.__module__.startswith(root_pkg):
            continue  # only classes defined in our package
        qual = f"{obj.__module__}.{name}"
        bases = tuple(
            f"{b.__module__}.{b.__name__}"
            for b in obj.__bases__
            if isinstance(b, type) and b.__module__.startswith(root_pkg)
        )
        doc = inspect.getdoc(obj) or ""
        out.append(ClassInfo(qualname=qual, modname=obj.__module__, name=name, bases=bases, doc=doc))


# ------------------------------ Graphviz backend -----------------------------

def render_graphviz(
    model: Model,
    out_path: str = "motion_kinematics/out/classes",
    fmt: str = "png",
    rankdir: str = "LR",
) -> str:
    """
    Render a class diagram using Graphviz (if installed).

    Parameters
    ----------
    model : Model
        Output of `discover()`.
    out_path : str
        Path without extension. We append .<fmt>.
    fmt : {'png','svg','pdf',...}
        Graphviz output format.
    rankdir : {'LR','TB'}
        Layout direction.

    Returns
    -------
    str
        The path_planning of the generated file.

    Raises
    ------
    RuntimeError if `graphviz` is not available.
    """
    try:
        from graphviz import Digraph  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Graphviz python package not installed. pip install graphviz") from exc

    dot = Digraph("classes", format=fmt)
    dot.attr(rankdir=rankdir, fontname="Helvetica", fontsize="11")
    dot.attr("node", shape="record", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    # Nodes
    for cls in model.classes:
        label = f"{{{cls.name}|{cls.modname}}}"
        dot.node(cls.qualname, label=label)

    # Edges (inheritance)
    for cls in model.classes:
        for base in cls.bases:
            dot.edge(base, cls.qualname, arrowhead="empty")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rendered = dot.render(out_path, cleanup=True)
    return rendered


# ------------------------------ pyreverse wrapper ----------------------------

def generate_pyreverse(package: str = "motion_kinematics", out_dir: str = "motion_kinematics/out", fmt: str = "png") -> List[str]:
    """
    Run pyreverse (from pylint) to generate UML diagrams.

    Returns list of produced files (usually classes.<fmt>, packages.<fmt>).

    Notes
    -----
    - Requires pylint to be installed and `pyreverse` available on PATH.
    - We move the resulting files into `out_dir`.
    """
    if shutil.which("pyreverse") is None:  # pragma: no cover
        raise RuntimeError("pyreverse not found on PATH. Install pylint: pip install pylint")

    os.makedirs(out_dir, exist_ok=True)

    cmd = ["pyreverse", "-o", fmt, "-p", package, package, "-A", "-S"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:  # pragma: no cover
        raise RuntimeError(f"pyreverse failed: {proc.stderr}")

    produced: List[str] = []
    for stem in ("classes", "packages"):
        src = f"{stem}.{fmt}"
        if os.path.exists(src):
            dst = os.path.join(out_dir, src)
            shutil.move(src, dst)
            produced.append(dst)
    return produced


# ------------------------------ Text emitters --------------------------------

def emit_plantuml(model: Model, path: str) -> str:
    """
    Emit a PlantUML class diagram (.puml) for the model.
    """
    lines: List[str] = ["@startuml", "skinparam classAttributeIconSize 0"]
    pkg = model.package

    # Classes
    for c in model.classes:
        lines.append(f'class "{c.name}" as {c.qualname} << (P,#FFCC00) {pkg} >> {{}}')

    # Inheritance
    for c in model.classes:
        for b in c.bases:
            lines.append(f"{b} <|-- {c.qualname}")

    lines.append("@enduml")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def emit_mermaid(model: Model, path: str) -> str:
    """
    Emit a Mermaid class diagram (.mmd) for the model.
    """
    lines: List[str] = ["classDiagram"]
    for c in model.classes:
        lines.append(f'class {c.qualname} {{}}')
    for c in model.classes:
        for b in c.bases:
            lines.append(f"{b} <|-- {c.qualname}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ------------------------------ JSON model -----------------------------------

def save_model_json(model: Model, path: str) -> str:
    """
    Save the discovered model as JSON (great for snapshot tests).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "package": model.package,
                "classes": [asdict(c) for c in model.classes],
            },
            f,
            indent=2,
        )
    return path


# ------------------------------ Tiny CLI (optional) --------------------------

def _build_cli():
    import argparse
    p = argparse.ArgumentParser(prog="motion_kinematics.design", description="Motion class diagram tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="Print JSON model to stdout")
    d.add_argument("--package", default="motion_kinematics")

    g = sub.add_parser("graphviz", help="Render with Graphviz")
    g.add_argument("--package", default="motion_kinematics")
    g.add_argument("--out", default="motion_kinematics/out/classes")
    g.add_argument("--fmt", default="png", choices=["png", "svg", "pdf"])
    g.add_argument("--rankdir", default="LR", choices=["LR", "TB"])

    r = sub.add_parser("pyreverse", help="Run pyreverse (pylint)")
    r.add_argument("--package", default="motion_kinematics")
    r.add_argument("--out", default="motion_kinematics/out")
    r.add_argument("--fmt", default="png")

    u = sub.add_parser("plantuml", help="Emit PlantUML file")
    u.add_argument("--package", default="motion_kinematics")
    u.add_argument("--out", default="motion_kinematics/out/classes.puml")

    m = sub.add_parser("mermaid", help="Emit Mermaid file")
    m.add_argument("--package", default="motion_kinematics")
    m.add_argument("--out", default="motion_kinematics/out/classes.mmd")

    j = sub.add_parser("json", help="Save JSON model")
    j.add_argument("--package", default="motion_kinematics")
    j.add_argument("--out", default="motion_kinematics/out/classes.json")

    return p


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.cmd == "discover":
        model = discover(args.package)
        print(json.dumps({"package": model.package, "classes": [asdict(c) for c in model.classes]}, indent=2))
        return

    if args.cmd in {"graphviz", "pyreverse", "plantuml", "mermaid", "json"}:
        model = discover(args.package)

    if args.cmd == "graphviz":
        path = render_graphviz(model, out_path=args.out, fmt=args.fmt, rankdir=args.rankdir)
        print(path); return

    if args.cmd == "pyreverse":
        paths = generate_pyreverse(args.package, out_dir=args.out, fmt=args.fmt)
        print("\n".join(paths)); return

    if args.cmd == "plantuml":
        print(emit_plantuml(model, args.out)); return

    if args.cmd == "mermaid":
        print(emit_mermaid(model, args.out)); return

    if args.cmd == "json":
        print(save_model_json(model, args.out)); return


if __name__ == "__main__":  # pragma: no cover
    main()
