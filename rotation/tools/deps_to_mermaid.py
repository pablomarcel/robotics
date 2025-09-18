# rotation/tools/deps_to_mermaid.py
from __future__ import annotations
import json, sys, pathlib

def to_mermaid(d, prefix="rotation"):
    lines = ["flowchart LR"]
    # pydeps --show-deps outputs a dict: {module: [deps...]}
    # pydeps --show-raw-deps may output {"imports": {module: [deps...]}, ...}
    if isinstance(d, dict) and "imports" in d and isinstance(d["imports"], dict):
        graph = d["imports"]
    else:
        graph = d if isinstance(d, dict) else {}

    def short(name: str) -> str:
        return name.replace(prefix + ".", "") if name.startswith(prefix + ".") else name

    seen = set()
    for src, deps in graph.items():
        if not isinstance(deps, (list, tuple)):
            continue
        for dst in deps:
            # keep it focused on your package
            if not (src.startswith(prefix) or dst.startswith(prefix)):
                continue
            edge = (short(src), short(dst))
            if edge not in seen:
                seen.add(edge)
                lines.append(f'  {edge[0]} --> {edge[1]}')
    return "\n".join(lines)

def main():
    if len(sys.argv) < 3:
        print("usage: python -m rotation.tools.deps_to_mermaid <in.json> <out.mmd>")
        sys.exit(2)
    inp, out = sys.argv[1], sys.argv[2]
    data = json.loads(pathlib.Path(inp).read_text())
    mm = to_mermaid(data, prefix="rotation")
    pathlib.Path(out).write_text(mm)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
