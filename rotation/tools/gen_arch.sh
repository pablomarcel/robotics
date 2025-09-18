#!/usr/bin/env bash
set -euo pipefail

# 1) Module deps JSON (no Graphviz output)
pydeps rotation --only rotation --noshow --no-output \
  --show-deps --deps-output rotation/out/arch.json

# 2) Convert deps JSON -> Mermaid
python -m rotation.tools.deps_to_mermaid rotation/out/arch.json > rotation/out/deps.mmd

# 3) Function call graph (AST -> Mermaid)
python -m rotation.tools.ast_callgraph rotation > rotation/out/callgraph_ast.mmd

echo "Wrote:"
echo " - rotation/out/arch.json"
echo " - rotation/out/deps.mmd (Mermaid)"
echo " - rotation/out/callgraph_ast.mmd (Mermaid)"
