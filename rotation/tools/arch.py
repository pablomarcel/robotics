# rotation/tools/arch.py
from __future__ import annotations
import ast, os, sys, argparse

PKG = "rotation"

def find_py_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)

def module_name_from_path(path: str) -> str:
    # e.g. rotation/rot_core.py -> rotation.rot_core
    rel = os.path.relpath(path).replace(os.sep, "/")
    if not rel.startswith(PKG + "/"):
        return None
    mod = rel[:-3]  # strip .py
    mod = mod.replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[:-9]
    return mod

def scan_imports(py_path: str):
    with open(py_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=py_path)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports

def build_graph(root: str):
    edges = []
    mods = []
    for path in find_py_files(root):
        m = module_name_from_path(path)
        if not m:
            continue
        mods.append(m)
        for imp in scan_imports(path):
            # Keep only intra-package edges
            if imp.startswith(PKG + "."):
                edges.append((m, imp))
    return sorted(set(mods)), sorted(set(edges))

def to_mermaid(mods, edges):
    lines = ["flowchart LR"]
    short = lambda s: s.replace(PKG + ".", "")
    for m in mods:
        lines.append(f"  {short(m)}")
    for a, b in edges:
        lines.append(f"  {short(a)} --> {short(b)}")
    return "\n".join(lines)

def to_ascii(mods, edges):
    # Simple adjacency list (no layout)
    short = lambda s: s.replace(PKG + ".", "")
    out = []
    adj = {m: [] for m in mods}
    for a,b in edges:
        adj[a].append(b)
    for m in mods:
        out.append(short(m))
        deps = sorted(adj[m])
        for d in deps:
            out.append(f"  -> {short(d)}")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser(description="Lightweight architecture graph")
    ap.add_argument("--format", choices=["mermaid","ascii"], default="ascii")
    ap.add_argument("--out", help="Output file (optional)")
    ap.add_argument("--root", default=PKG, help="Package root (default: rotation)")
    args = ap.parse_args()

    mods, edges = build_graph(args.root)
    text = to_mermaid(mods, edges) if args.format == "mermaid" else to_ascii(mods, edges)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)

if __name__ == "__main__":
    main()
