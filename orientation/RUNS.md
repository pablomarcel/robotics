# Orientation Kinematics — CLI Runbook

This bundle contains an exhaustive list of run commands for the **object‑oriented** Orientation toolkit you’ve built.

The CLI lives at `orientation/cli.py` and exposes **file-friendly** verbs so you don’t have to fight shell parsing.
It also keeps results in `orientation/out/` and reads inputs from `orientation/in/` when you use the batch runner.

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
runroot mkdir -p orientation/out
# -----------------------------------------------------------------------------
```

---

## Quick sanity checks

```bash
# help
runroot python -m orientation.cli -h
```

```bash
runroot python -m orientation.cli matrix-from-axis -h
```

---

## Core conversions / ops

### 1) Axis–angle → matrix
```bash
runroot python -m orientation.cli matrix-from-axis --axis 0 0 1 --phi 1.57079632679
```

```bash
runroot python -m orientation.cli matrix-from-axis --axis 1 0 0 --phi 0.3 --save R_x.csv
```

### 2) Compose two axis–angle turns (R2 * R1)
```bash
runroot python -m orientation.cli compose-axis --phi1 0.4 --axis1 1 0 0 --phi2 0.7 --axis2 0 0 1 --save R_comp.csv
```

### 3) Matrix → quaternion (Euler parameters e0 e1 e2 e3)
```bash
runroot python -m orientation.cli to-quat --matrix 1 0 0  0 1 0  0 0 1
```

### 4) Quaternion → matrix
```bash
runroot python -m orientation.cli from-quat --quat 0.9238795 0 0.3826834 0 --save R_q.csv
```

### 5) Rodrigues vector → matrix
```bash
runroot python -m orientation.cli rodrigues-to-matrix --w 0.1 0.2 0.3 --save R_rod.csv
```

### 6) Matrix → Rodrigues vector
```bash
runroot python -m orientation.cli matrix-to-rodrigues --matrix 1 0 0  0 1 0  0 0 1
```

### 7) Euler angles → matrix (order configurable)
```bash
runroot python -m orientation.cli euler-to-matrix --angles 0.3 0.2 0.1 --order ZYX
```

```bash
runroot python -m orientation.cli euler-to-matrix --angles 10 5 2 --order XYZ --deg --save R_eul.csv
```

### 8) Matrix → Euler angles
```bash
runroot python -m orientation.cli matrix-to-euler --matrix 0.936293 0.289629 -0.198669 -0.275096 0.957826 0.077458 0.218351 0.0 0.975870 --order ZYX
```

```bash
runroot python -m orientation.cli matrix-to-euler --matrix 1 0 0  0 0 -1  0 1 0 --order ZXZ --deg
```

### 9) Exponential map: `exp(omega^)`
```bash
runroot python -m orientation.cli expmap --omega 0.2 0.0 0.0 --save R_exp.csv
```

### 10) Random SO(3) sampling
```bash
runroot python -m orientation.cli random-so3 --n 5
```

```bash
runroot python -m orientation.cli random-so3 --n 100 --out random_100.json
```

---

## File‑based helpers (no shell quoting headaches)

These variants read matrices from files:

- CSV: 3×3, **row‑major**, commas (see `samples/matrix_I.csv`)
- JSON: either
  - `{"matrix": [[...],[...],[...]]}` or
  - `{"matrix": [r11,r12,...,r33]}`

### 11) Matrix → quaternion (file)
```bash
runroot python -m orientation.cli to-quat-file --in samples/matrix_example.csv
```

```bash
runroot python -m orientation.cli to-quat-file --in samples/matrix_I.csv
```

### 12) Matrix → Rodrigues (file)
```bash
runroot python -m orientation.cli matrix-to-rodrigues-file --in samples/matrix_I.csv
```

*(If you also need a “matrix-from-file” verb for bulk processing, we can add it too.)*

---

## Batch runner (reads/writes under orientation/in & out)

Put a JSON list of jobs under `orientation/in/` and run:
```bash
cp samples/jobs.json orientation/in/jobs.json
```

```bash
runroot python -m orientation.cli batch --in jobs.json --out results.json
```

The batch file supports the same ops as the CLI:
`matrix-from-axis`, `compose-axis`, `to-quat`, `from-quat`,
`rodrigues-to-matrix`, `matrix-to-rodrigues`, `euler-to-matrix`, `expmap`.

---

## Class diagrams (no local PlantUML required)

### Option A — Standalone diagram tool (recommended)
```bash
runroot python -m orientation.tools.gen_diagram              # writes orientation/out/class_diagram.puml
```

Render to PNG via Kroki (no installs):
```bash
curl -s -H "Content-Type: text/plain"   --data-binary @orientation/out/class_diagram.puml   https://kroki.io/plantuml/png > orientation/out/class_diagram.png
```

### Option B — Multi-format via design.py
```bash
runroot python - <<'PY'
from pathlib import Path
from orientation.design import generate_diagram
generate_diagram(Path("orientation/out"))
print("wrote .puml, .dot, .mmd")
PY

# Render any of them with Kroki:
curl -s -H "Content-Type: text/plain"   --data-binary @orientation/out/class_diagram.mmd   https://kroki.io/mermaid/png > orientation/out/class_diagram_mermaid.png
```

---

## Tips for zsh users

- Prefer the **file‑based** verbs to avoid word-splitting issues.
- If you really want to inline a CSV file into `--matrix`, scrub commas and newlines:
  ```bash
  runroot python -m orientation.cli to-quat --matrix $(tr ',\n' '  ' < orientation/out/R_comp.csv | xargs)
  ```
  but again: file-based is cleaner.

---

## Samples included in this bundle

- `samples/matrix_I.csv`
- `samples/matrix_example.csv`
- `samples/quat_example.json`
- `samples/rodrigues_example.json`
- `samples/jobs.json`

Use them as templates for your own inputs under `orientation/in/`.

---
