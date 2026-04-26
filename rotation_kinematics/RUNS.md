
# Rotation Kinematics — Run Commands (Exhaustive)

This README collects repeatable commands for **running the CLI**, **exporting diagrams**, and **generating architecture/call graphs & traces** for your `rotation` module.

> All paths below assume you run the commands from the project root (where the `rotation/` package lives).  
> Output is written to `rotation/out/` (created if missing).  
> Replace values (angles, files) as needed.

---

## -1) One-time session bootstrap (copy/paste once per new shell)
```bash
# --- run-from-root helpers ----------------------------------------------------
# Find project root: prefer Git; otherwise, walk up until we see a marker file.
_mc_root() {
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse --show-toplevel
    return
  fi
  # Fallback: ascend until we find a recognizable root marker.
  local d="$PWD"
  while [ "$d" != "/" ]; do
    if [ -d "$d/.git" ] || [ -f "$d/pytest.ini" ] || [ -f "$d/pyproject.toml" ]; then
      echo "$d"; return
    fi
    d="$(dirname "$d")"
  done
  echo "$PWD"
}

# Run a command from the project root (without changing your current shell dir)
runroot() { ( cd "$(_mc_root)" && "$@" ); }

# Ensure out/ exists where the app expects to write
runroot mkdir -p rotation_kinematics/out
# -----------------------------------------------------------------------------
```

## 0) Quick setup

```bash
# Activate your venv first (if not already)
source .venv/bin/activate  # or the equivalent on your system

# Ensure output/input dirs exist
runroot python - <<'PY'
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
runroot python -m rotation_kinematics.rot_cli compose local zyz "10,20,30" --degrees
```

# or

```bash
rot-cli compose local zyz "10,20,30" --degrees
```

### Decompose a matrix back to angles
```bash
# Compose then decompose (degrees)
runroot python -m rotation_kinematics.rot_cli decompose local zyz --from-angles "10,20,30" --degrees
```

# Decompose from CSV
```bash
runroot python -m rotation_kinematics.rot_cli decompose global zyx --from-csv rotation_kinematics/in/R.csv
```

### Active transform of points
```bash
# Inline points: "x;y;z|x;y;z|..."
runroot python -m rotation_kinematics.rot_cli transform local zyz "10,20,30"   --degrees --points "1;0;0|0;1;0|0;0;1"
```

# From CSV under rotation/in/
```bash
runroot python -m rotation_kinematics.rot_cli transform local zyz "10,20,30"   --degrees --points basis.csv --save P_transformed.csv
```

### Passive transform (coordinate change)
```bash
runroot python -m rotation_kinematics.rot_cli passive local zyz "10,20,30" --degrees   --points "1;0;0|0;1;0|0;0;1"
```

### Repeat/exponentiate a rotation
```bash
runroot python -m rotation_kinematics.rot_cli repeat local zyz "10,20,30" 12 --degrees
```

### Align body x-axis to a vector
```bash
runroot python -m rotation_kinematics.rot_cli align "0.2, 0.3, 0.9"
```

### Orthogonality check
```bash
runroot python -m rotation_kinematics.rot_cli check --from-angles "10,20,30"   --mode local --seq zyz --degrees
# Or from CSV:
```

```bash
runroot python -m rotation_kinematics.rot_cli check --from-csv rotation_kinematics/in/R.csv
```

### Closed-form `E(q)` matrices (symbolic)
```bash
# General proper/TB sequences
runroot python -m rotation_kinematics.rot_cli E zyz --convention local --frame body
```

# RPY convenience (ZYX) + RPY column reordering
```bash
runroot python -m rotation_kinematics.rot_cli E zyx --convention local --frame body --rpy-order
```

### ω ↔ q̇ mappings
```bash
# ω from rates (units follow --degrees)
runroot python -m rotation_kinematics.rot_cli angvel local zyz "10,20,30" "0.1,0.2,0.3"   --degrees --frame body
```

# rates from ω
```bash
runroot python -m rotation_kinematics.rot_cli rates local zyz "10,20,30" "0.7,0.2,0.1"   --degrees --frame body
```

### Batch jobs (JSON/YAML or simple key=value)
```bash
# rotation_kinematics/in/job.yaml must define a top-level 'tasks' list
runroot python -m rotation_kinematics.rot_cli batch job.yaml
```

### Dump runtime-traced edges (from decorators)
```bash
# From the CLI itself
runroot python -m rotation_kinematics.rot_cli uml --out rotation_kinematics/out/runtime_sequence.puml
```

---

## 2) Composition Root & Class Diagrams (module: `rotation.app`)

Exports the Facade + APIs class diagram, and dumps any runtime trace captured by decorators.

```bash
# PlantUML
runroot python -m rotation_kinematics.app puml    -o rotation_kinematics/out/app_class.puml
```

# Mermaid
```bash
runroot python -m rotation_kinematics.app mermaid -o rotation_kinematics/out/app_class.mmd
```

# Dump runtime sequence edges (same sink the CLI uses)
```bash
runroot python -m rotation_kinematics.app runtime -o rotation_kinematics/out/runtime_sequence.puml
```

---

## 3) Dependency graph (imports) **without** Graphviz `dot`

We use `pydeps` to write JSON and our helper to convert to Mermaid.

```bash
# JSON of intra-package imports
pydeps rotation_kinematics --only rotation_kinematics --noshow --no-output   --deps-output rotation_kinematics/out/arch.json
```

```bash
# Convert to Mermaid (requires rotation_kinematics/tools/deps_to_mermaid.py)
runroot python -m rotation_kinematics.tools.deps_to_mermaid rotation_kinematics/out/arch.json   > out/deps_imports.mmd
```

Optional ASCII import listing (pure runroot python):
```bash
runroot python -m rotation_kinematics.tools.arch_ascii rotation_kinematics > rotation_kinematics/out/imports_ascii.txt
```

---

## 4) Static call graph (AST-based, no execution)

```bash
# Writes a Mermaid flowchart of static call edges it can infer
runroot python -m rotation_kinematics.tools.ast_callgraph rotation_kinematics   > out/callgraph_ast.mmd
```

---

## 5) Runtime tracing & flamegraphs (VizTracer)

Profile one CLI run and view in the browser or export HTML.

```bash
# JSON trace
runroot viztracer -o rotation_kinematics/out/trace.json -m rotation_kinematics.rot_cli   angvel local zyz "10,20,30" "0.1,0.2,0.3" --degrees --frame body
```

# Open viewer (served on http://localhost:9001)
```bash
runroot vizviewer rotation_kinematics/out/trace.json
```

# Self-contained HTML report
```bash
runroot viztracer -o rotation_kinematics/out/trace.html --html -m rotation_kinematics.rot_cli   compose local zyz "10,20,30" --degrees
```

**Tip:** to include more/less detail:
```bash
runroot viztracer -o rotation_kinematics/out/trace.json   --min_duration 0.0005 --max_stack_depth 20   -m rotation_kinematics.rot_cli compose local zyz "10,20,30" --degrees
```

---

## 6) PyCharm diagram quick notes (FYI)

- Right-click a symbol (class/function/file) → **Diagrams** → **Show Diagram…** → runroot python.
- Use toolbar to expand related classes / show method signatures.
- This is IDE-only; for shareable artifacts, prefer the exporters above.

---

## 7) Troubleshooting

- **`pydeps: cannot find 'dot'`** — use the JSON+Mermaid path shown above (no Graphviz needed).
- **`viztracer ... invalid choice: 'runroot python'`** — when using `viztracer`, prefer the `-m package.module` form (see examples).
- **Missing output dirs** — they are auto-created by the CLI and exporters, but you can pre-create with:
  ```bash
  mkdir -p rotation_kinematics/out rotation_kinematics/in
  ```
- **Unicode in terminal** — Some outputs print ω, φ, θ; ensure your terminal’s UTF‑8 encoding is set.

---

## 8) One-liners you might copy/paste a lot

```bash
# Class diagrams (both formats)
runroot python -m rotation_kinematics.app puml -o rotation_kinematics/out/app_class.puml && runroot python -m rotation_kinematics.app mermaid -o rotation_kinematics/out/app_class.mmd

# Import graph (Mermaid)
pydeps rotation_kinematics --only rotation_kinematics --noshow --no-output   --deps-output rotation_kinematics/out/arch.json && runroot python -m rotation_kinematics.tools.deps_to_mermaid rotation_kinematics/out/arch.json   > rotation_kinematics/out/deps_imports.mmd

# Static call graph
runroot python -m rotation_kinematics.tools.ast_callgraph rotation_kinematics   > rotation_kinematics/out/callgraph_ast.mmd

# Quick ω from rates demo + trace
viztracer -o rotation_kinematics/out/trace.json -m rotation_kinematics.rot_cli   angvel local zyz "10,20,30" "0.1,0.2,0.3" --degrees --frame body
```

### Sphinx

python -m rotation_kinematics.rot_cli sphinx-skel rotation_kinematics/docs
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html
sphinx-autobuild docs docs/_build/html