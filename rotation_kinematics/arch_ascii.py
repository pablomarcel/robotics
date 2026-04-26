# rotation_kinematics/arch_ascii.py
from __future__ import annotations
import ast, os, pathlib
from collections import defaultdict

PKG = "rotation_kinematics"
FILES = ["rot_cli.py","rot_core.py","rot_design.py","rot_io.py","rot_utils.py"]

def find_internal_imports(py_path: str) -> set[str]:
    src = pathlib.Path(py_path).read_text(encoding="utf-8")
    tree = ast.parse(src, filename=py_path)
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.startswith(PKG + "."):
                    used.add(n.name)
                elif n.name in {PKG}:
                    used.add(PKG)
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "")
            if mod.startswith(PKG + "."):
                used.add(mod)
            elif mod == PKG:
                used.add(PKG)
    return used

def main():
    root = pathlib.Path(__file__).resolve().parent
    edges = defaultdict(set)
    modules = []
    for f in FILES:
        mod = f"{PKG}.{pathlib.Path(f).stem}"
        modules.append(mod)
        deps = find_internal_imports(root / f)
        for d in deps:
            edges[mod].add(d)

    # Print simple ASCII graph
    print(f"Internal import graph for '{PKG}':\n")
    for m in sorted(modules):
        print(f"└─ {m}")
        deps = sorted(d for d in edges[m] if d != m)
        for i, d in enumerate(deps):
            branch = "   ├─" if i < len(deps)-1 else "   └─"
            print(f"{branch} {d}")

if __name__ == "__main__":
    main()
