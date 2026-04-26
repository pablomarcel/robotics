from __future__ import annotations
import ast, sys, pathlib, os
from collections import defaultdict

def find_calls_in_file(path: str):
    src = pathlib.Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src, filename=path)
    module = pathlib.Path(path).stem
    parents = []  # stack of qualnames
    edges = set()
    defs  = set()

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            qn = f"{module}.{node.name}"
            defs.add(qn)
            parents.append(qn)
            self.generic_visit(node)
            parents.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            # include class as a scope; methods become module.Class.method
            qn = f"{module}.{node.name}"
            parents.append(qn)
            self.generic_visit(node)
            parents.pop()

        def visit_Call(self, node: ast.Call):
            # best-effort callee name (foo(), mod.bar())
            callee = None
            if isinstance(node.func, ast.Name):
                callee = f"{module}.{node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                # try to resolve "mod.func" to a dotted string
                parts = []
                cur = node.func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                    parts.reverse()
                    callee = ".".join(parts)
                    # If it’s local attribute call like self.x, still keep the attr name
                    if parts[0] in {"self", "cls"}:
                        callee = f"{module}." + ".".join(parts[1:])
                else:
                    callee = f"{module}.<dynamic>"
            else:
                callee = f"{module}.<dynamic>"

            caller = parents[-1] if parents else f"{module}.__module__"
            edges.add((caller, callee))
            self.generic_visit(node)

    V().visit(tree)
    return defs, edges

def to_mermaid(edges: set[tuple[str, str]]):
    # Build a Mermaid flowchart
    lines = ["flowchart LR"]
    for a, b in sorted(edges):
        # sanitize node ids for Mermaid (no spaces)
        aid = a.replace(".", "_")
        bid = b.replace(".", "_")
        lines.append(f'  {aid}["{a}"] --> {bid}["{b}"]')
    return "\n".join(lines)

def main():
    if len(sys.argv) < 3:
        print("usage: python -m rotation_kinematics.tools.ast_callgraph <glob or paths...> <out.mmd>")
        sys.exit(2)

    *paths, out = sys.argv[1:]
    files = []
    for p in paths:
        pth = pathlib.Path(p)
        if pth.is_dir():
            files += [str(x) for x in pth.rglob("*.py")]
        elif any(ch in p for ch in "*?[]"):
            files += [str(x) for x in pathlib.Path(".").glob(p)]
        else:
            files.append(str(pth))

    all_edges = set()
    for f in files:
        if os.path.basename(f).startswith("_"):  # skip __init__.py by default
            continue
        _, edges = find_calls_in_file(f)
        all_edges |= edges

    mm = to_mermaid(all_edges)
    pathlib.Path(out).write_text(mm, encoding="utf-8")
    print(f"wrote {out} ({len(all_edges)} edges)")

if __name__ == "__main__":
    main()
