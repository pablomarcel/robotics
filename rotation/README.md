
# Rotation Kinematics — Run Commands (Exhaustive)

This README collects repeatable commands for **running the CLI**, **exporting diagrams**, and **generating architecture/call graphs & traces** for your `rotation` module.

> All paths below assume you run the commands from the project root (where the `rotation/` package lives).  
> Output is written to `rotation/out/` (created if missing).  
> Replace values (angles, files) as needed.

---

## 0) Quick setup

```bash
# Activate your venv first (if not already)
source .venv/bin/activate  # or the equivalent on your system

# Ensure output/input dirs exist
python - <<'PY'
import os
for d in ["rotation/out","rotation/in"]:
    os.makedirs(d, exist_ok=True)
print("ok")
PY
```

---

## 1) Core CLI (module: `rotation.rot_cli`, prog name: `rot-cli`)

You can invoke via module or the console script (if installed as a package).

### Compose a rotation matrix
```bash
python -m rotation.rot_cli compose local zyz "10,20,30" --degrees
# or
rot-cli compose local zyz "10,20,30" --degrees
```

### Decompose a matrix back to angles
```bash
# Compose then decompose (degrees)
python -m rotation.rot_cli decompose local zyz --from-angles "10,20,30" --degrees

# Decompose from CSV
python -m rotation.rot_cli decompose global zyx --from-csv rotation/in/R.csv
```

### Active transform of points
```bash
# Inline points: "x;y;z|x;y;z|..."
python -m rotation.rot_cli transform local zyz "10,20,30"   --degrees --points "1;0;0|0;1;0|0;0;1"

# From CSV under rotation/in/
python -m rotation.rot_cli transform local zyz "10,20,30"   --degrees --points basis.csv --save P_transformed.csv
```

### Passive transform (coordinate change)
```bash
python -m rotation.rot_cli passive local zyz "10,20,30" --degrees   --points "1;0;0|0;1;0|0;0;1"
```

### Repeat/exponentiate a rotation
```bash
python -m rotation.rot_cli repeat local zyz "10,20,30" 12 --degrees
```

### Align body x-axis to a vector
```bash
python -m rotation.rot_cli align "0.2, 0.3, 0.9"
```

### Orthogonality check
```bash
python -m rotation.rot_cli check --from-angles "10,20,30"   --mode local --seq zyz --degrees
# Or from CSV:
python -m rotation.rot_cli check --from-csv rotation/in/R.csv
```

### Closed-form `E(q)` matrices (symbolic)
```bash
# General proper/TB sequences
python -m rotation.rot_cli E zyz --convention local --frame body

# RPY convenience (ZYX) + RPY column reordering
python -m rotation.rot_cli E zyx --convention local --frame body --rpy-order
```

### ω ↔ q̇ mappings
```bash
# ω from rates (units follow --degrees)
python -m rotation.rot_cli angvel local zyz "10,20,30" "0.1,0.2,0.3"   --degrees --frame body

# rates from ω
python -m rotation.rot_cli rates local zyz "10,20,30" "0.7,0.2,0.1"   --degrees --frame body
```

### Batch jobs (JSON/YAML or simple key=value)
```bash
# rotation/in/job.yaml must define a top-level 'tasks' list
python -m rotation.rot_cli batch job.yaml
```

### Dump runtime-traced edges (from decorators)
```bash
# From the CLI itself
python -m rotation.rot_cli uml --out rotation/out/runtime_sequence.puml
```

---

## 2) Composition Root & Class Diagrams (module: `rotation.app`)

Exports the Facade + APIs class diagram, and dumps any runtime trace captured by decorators.

```bash
# PlantUML
python -m rotation.app puml    -o rotation/out/app_class.puml

# Mermaid
python -m rotation.app mermaid -o rotation/out/app_class.mmd

# Dump runtime sequence edges (same sink the CLI uses)
python -m rotation.app runtime -o rotation/out/runtime_sequence.puml
```

---

## 3) Dependency graph (imports) **without** Graphviz `dot`

We use `pydeps` to write JSON and our helper to convert to Mermaid.

```bash
# JSON of intra-package imports
pydeps rotation --only rotation --noshow --no-output   --deps-output rotation/out/arch.json

# Convert to Mermaid (requires rotation/tools/deps_to_mermaid.py)
python -m rotation.tools.deps_to_mermaid rotation/out/arch.json   > rotation/out/deps_imports.mmd
```

Optional ASCII import listing (pure Python):
```bash
python -m rotation.tools.arch_ascii rotation > rotation/out/imports_ascii.txt
```

---

## 4) Static call graph (AST-based, no execution)

```bash
# Writes a Mermaid flowchart of static call edges it can infer
python -m rotation.tools.ast_callgraph rotation   > rotation/out/callgraph_ast.mmd
```

---

## 5) Runtime tracing & flamegraphs (VizTracer)

Profile one CLI run and view in the browser or export HTML.

```bash
# JSON trace
viztracer -o rotation/out/trace.json -m rotation.rot_cli   angvel local zyz "10,20,30" "0.1,0.2,0.3" --degrees --frame body

# Open viewer (served on http://localhost:9001)
vizviewer rotation/out/trace.json

# Self-contained HTML report
viztracer -o rotation/out/trace.html --html -m rotation.rot_cli   compose local zyz "10,20,30" --degrees
```

**Tip:** to include more/less detail:
```bash
viztracer -o rotation/out/trace.json   --min_duration 0.0005 --max_stack_depth 20   -m rotation.rot_cli compose local zyz "10,20,30" --degrees
```

---

## 6) PyCharm diagram quick notes (FYI)

- Right-click a symbol (class/function/file) → **Diagrams** → **Show Diagram…** → Python.
- Use toolbar to expand related classes / show method signatures.
- This is IDE-only; for shareable artifacts, prefer the exporters above.

---

## 7) Troubleshooting

- **`pydeps: cannot find 'dot'`** — use the JSON+Mermaid path shown above (no Graphviz needed).
- **`viztracer ... invalid choice: 'python'`** — when using `viztracer`, prefer the `-m package.module` form (see examples).
- **Missing output dirs** — they are auto-created by the CLI and exporters, but you can pre-create with:
  ```bash
  mkdir -p rotation/out rotation/in
  ```
- **Unicode in terminal** — Some outputs print ω, φ, θ; ensure your terminal’s UTF‑8 encoding is set.

---

## 8) One-liners you might copy/paste a lot

```bash
# Class diagrams (both formats)
python -m rotation.app puml -o rotation/out/app_class.puml && python -m rotation.app mermaid -o rotation/out/app_class.mmd

# Import graph (Mermaid)
pydeps rotation --only rotation --noshow --no-output   --deps-output rotation/out/arch.json && python -m rotation.tools.deps_to_mermaid rotation/out/arch.json   > rotation/out/deps_imports.mmd

# Static call graph
python -m rotation.tools.ast_callgraph rotation   > rotation/out/callgraph_ast.mmd

# Quick ω from rates demo + trace
viztracer -o rotation/out/trace.json -m rotation.rot_cli   angvel local zyz "10,20,30" "0.1,0.2,0.3" --degrees --frame body
```
