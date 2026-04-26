#!/usr/bin/env bash
set -euo pipefail

# 1) Module deps JSON (no Graphviz output)
pydeps rotation_kinematics --only rotation_kinematics --noshow --no-output \
  --show-deps --deps-output rotation_kinematics/out/arch.json

# 2) Convert deps JSON -> Mermaid
python -m rotation_kinematics.tools.deps_to_mermaid rotation_kinematics/out/arch.json > rotation_kinematics/out/deps.mmd

# 3) Function call graph (AST -> Mermaid)
python -m rotation_kinematics.tools.ast_callgraph rotation_kinematics > rotation_kinematics/out/callgraph_ast.mmd

echo "Wrote:"
echo " - rotation/out/arch.json"
echo " - rotation/out/deps.mmd (Mermaid)"
echo " - rotation/out/callgraph_ast.mmd (Mermaid)"
