# Robot Dynamics App — RUNS.md (OOP + TDD)

> Copy these commands and run them from your project root (the directory that contains the `robot/` package).
> All outputs go to `robot/out/`. Inputs are read from `robot/in/`.

**Updated:** 2025-10-12T20:21:37

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
runroot mkdir -p robot_dynamics/out
# -----------------------------------------------------------------------------
```

---

## 0) Environment & install

```bash
# (recommended) create venv
runroot python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# base deps
pip install -U pip wheel
pip install numpy sympy pyyaml pytest

# optional extras
pip install pylint            # for pyreverse class diagram
pip install pinocchio         # optional: real-time rigid-body dynamics (engine stub ready)
pip install pytest-cov        # if you want coverage
```

---

## 1) Quick sanity / help

```bash
runroot python -m robot_dynamics.cli --help
```

```bash
runroot python -m robot_dynamics.tools.uml --help
```

---

## 2) Run tests (TDD)

```bash
# minimal
pytest robot_dynamics/tests -q

# verbose with timings
pytest robot_dynamics/tests -q -s -vv

# coverage (if pytest-cov installed)
pytest robot_dynamics/tests --cov=robot_dynamics --cov-report=term-missing
```

---

## 3) Built‑in example: **Planar 2R** (symbolic Lagrange engine)

> Computes **M(q), C(q, q̇), g(q), τ** and writes JSON to `robot/out/2r_result.json` by default.

### 3.1 Default run (mid‑link COMs, unit lengths/masses)
```bash
runroot python -m robot_dynamics.cli planar2r
```

### 3.2 Specify geometry & inertial params
```bash
runroot python -m robot_dynamics.cli planar2r   --l1 1.2 --l2 0.8   --m1 2.0 --m2 1.5
```

### 3.3 Choose engine (symbolic or pinocchio‑stub)
```bash
runroot python -m robot_dynamics.cli planar2r --engine sympy
```

```bash
runroot python -m robot_dynamics.cli planar2r --engine pinocchio
```

### 3.4 Set state (q, q̇, q̈) and gravity
```bash
runroot python -m robot_dynamics.cli planar2r   --q 0.2 -0.3   --qd 0.1 0.05   --qdd 0.0 0.0   --g 9.81
```

### 3.5 Save to a custom path (creates directories)
```bash
runroot python -m robot_dynamics.cli planar2r --out robot_dynamics/out/runs/planar2r_run1.json
```

---

## 4) Generic model from YAML (`robot/in/…`)

> Load `robot/in/<name>.yaml`, compute dynamics, and save a JSON peer next to it in `robot/out/`.

### 4.1 Use the included sample
```bash
# file: robot_dynamics/in/sample_2r.yaml
runroot python -m robot_dynamics.cli from-yaml sample_2r.yaml --engine sympy --q 0.2 -0.3 --qd 0.1 0.05 --qdd 0 0 --g 9.81
```

### 4.2 Sympy vs Pinocchio engine
```bash
runroot python -m robot_dynamics.cli from-yaml sample_2r.yaml --engine sympy   --q 0.1 0.2 --qd 0.0 0.0 --qdd 0 0
```

```bash
runroot python -m robot_dynamics.cli from-yaml sample_2r.yaml --engine pinocchio --q 0.1 0.2 --qd 0.0 0.0 --qdd 0 0
```

### 4.3 Different gravity direction/magnitude
```bash
# Moon gravity
runroot python -m robot_dynamics.cli from-yaml sample_2r.yaml --engine sympy --q 0.5 0.1 --qd 0.0 0.0 --qdd 0 0 --g 1.62
```

---

## 5) CLI flag matrix (exhaustive combinations)

### Subcommand: `planar2r`
- `--l1 FLOAT`  (default: `1.0`)
- `--l2 FLOAT`  (default: `1.0`)
- `--m1 FLOAT`  (default: `1.0`)
- `--m2 FLOAT`  (default: `1.0`)
- `--engine [sympy|pinocchio]`  (default: `sympy`)
- `--q FLOAT FLOAT`  (default: `0.2 0.3`)
- `--qd FLOAT FLOAT` (default: `0.1 -0.2`)
- `--qdd FLOAT FLOAT` (default: `0.0 0.0`)
- `--g FLOAT` (default: `9.81`)
- `--out PATH` (default: `robot/out/2r_result.json`)

**Examples:**
```bash
# (A) No accelerations provided -> τ computed with q̈=0
runroot python -m robot_dynamics.cli planar2r --q 0.2 0.3 --qd 0.1 -0.2 --g 9.81
```

# (B) With accelerations
```bash
runroot python -m robot_dynamics.cli planar2r --q 0.2 0.3 --qd 0.1 -0.2 --qdd 0.5 -0.1
```

# (C) Heavy link 2
```bash
runroot python -m robot_dynamics.cli planar2r --m2 5.0 --q 0.2 -0.3 --qd 0.15 0.0 --qdd 0 0
```

### Subcommand: `from-yaml`
- `name` (positional): YAML filename under `robot/in/`
- `--engine [sympy|pinocchio]` (default: `sympy`)
- `--q FLOAT ...`   (n values; must match DoF)
- `--qd FLOAT ...`  (n values; must match DoF)
- `--qdd FLOAT ...` (n values; optional; defaults to zeros if omitted)
- `--g FLOAT`  (default: `9.81`)

**Examples:**
```bash
# (D) Provide q, q̇ only (statics torque if q̇=0, q̈=0)
runroot python -m robot_dynamics.cli from-yaml sample_2r.yaml --q 0.5 -0.2 --qd 0 0
```

# (E) Dynamic state
```bash
runroot python -m robot_dynamics.cli from-yaml sample_2r.yaml --q 0.3 0.1 --qd 0.2 -0.1 --qdd 1.0 -0.4 --g 9.81
```

---

## 6) Outputs

Each run writes a JSON with some or all keys:
```json

```

---

## 7) UML class diagram (cute & sparkling)

> Requires `pylint` (for `pyreverse`). PNGs will be saved to `robot/out/uml/`.

```bash
runroot python -m robot_dynamics.tools.uml --engine pyreverse --out robot_dynamics/out/uml
```

```bash
runroot open robot_dynamics/out/uml/classes.png   # macOS; Linux: xdg-open
```

---

## 8) Developer flows

### 8.1 Run package as a module
```bash
runroot python -m robot_dynamics.app planar2r --q 0.2 -0.3 --qd 0.1 0.05 --qdd 0 0
```

### 8.2 Lint (optional)
```bash
ruff check robot_dynamics || true
pylint robot_dynamics || true
```

---

## 9) File conventions

- **Inputs** `robot/in/*.yaml` — robot definitions (links/joints); see `sample_2r.yaml`.
- **Outputs** `robot/out/*.json` — numerical results from runs.
- **Tests** `robot/tests/*` — pytest-based TDD suite.
- **Tools** `robot/tools/*` — helpers (e.g., UML diagram generator).

---

## 10) Troubleshooting

- `ImportError: PinocchioEngine` — engine is stubbed; install `pinocchio` only if you plan to implement a real URDF-backed pipeline.
- Shape errors: ensure `--q`, `--qd`, and `--qdd` lengths match the robot DoF.
- Floating-point mismatches: use `--g` consistent with your coordinate convention; by default +y is upward in the symbolic reference.

Happy hacking 🤖✨

### Sphinx

python -m robot_dynamics.cli sphinx-skel robot_dynamics/docs
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html
sphinx-autobuild docs docs/_build/html