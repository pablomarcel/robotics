# Time-Optimal Control (timeopt) — Runbook & Command Cheatsheet

This README is a **one-stop runbook** for your Robotics **time-optimal control** app (package: `timeopt`).  
It covers **installation**, **tests**, **CLI** usage, **examples** (CasADi & TOPPRA), **class diagram generation**, and **docs**.

> You’re on macOS with Python 3.11, `casadi==3.7.2`, `toppra==0.6.3`. Ipopt is **optional**; we auto-fallback to CasADi’s built‑in SQP.

---

## 0) Project Layout (relevant pieces)

```
robotics/
├─ timeopt/
│  ├─ __init__.py
│  ├─ app.py          # MinTimeDoubleIntegrator + TwoRPathTimeScaler (TOPPRA 0.6.3 API)
│  ├─ core.py         # Base classes + SolveResult
│  ├─ design.py       # 2R geometry helpers
│  ├─ cli.py          # (if present) CLI entry-points
│  └─ tests/
│     ├─ test_cli_smoke.py
│     ├─ test_core_double_integrator.py
│     └─ test_toppraline.py
└─ ...
```

If your CLI module lives elsewhere (`time/cli.py` etc.), adjust the `-m <module>` paths accordingly.

---

## 1) Environment & Installation

### 1.1 Create / activate venv
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -V
```

### 1.2 Install dependencies (minimum to run tests)
```bash
python -m pip install --upgrade pip
pip install casadi==3.7.2 toppra==0.6.3 numpy scipy pytest pytest-sugar
```
**Optional** (already in your env, but included for completeness):
```bash
pip install graphviz py2puml plantuml sphinx furo
```

> **Note on Ipopt**: If Ipopt is available, CasADi will use it. If not, our code falls back to `sqpmethod` automatically—no extra setup required.

---

## 2) Run the Test Suite (pytest)

From repo root:
```bash
pytest timeopt/tests -q
# or verbose:
pytest timeopt/tests -vv
# with coverage (if pytest-cov installed):
pytest timeopt/tests -q --cov=timeopt --cov-report=term-missing
```

Expected passing tests:
- `test_cli_smoke.py::test_diagram_cli_smoke`
- `test_core_double_integrator.py::test_double_integrator_sanity`
- `test_toppraline.py::test_toppraline_runs_and_returns_tf`

---

## 3) CLI Usage (if `timeopt/cli.py` exists)

### 3.1 Discover commands
```bash
python -m timeopt.cli --help
```

### 3.2 Generate class diagram via CLI
```bash
# Write a PlantUML (.puml) to time/out and render to PNG
python -m timeopt.cli diagram --package timeopt --out time/out/timeopt.puml
plantuml -tpng time/out/timeopt.puml -o .
# Result: time/out/timeopt.png
```

### 3.3 Run the double integrator (print JSON)
```bash
python -m timeopt.cli run double-integrator \
  --name di_cli --x0 0.0 --xf 1.0 --m 1.0 --F 10.0 \
  --mu 0.0 --drag 0.0 --g 9.81
```

### 3.4 Run the 2R TOPPRA time-scaling along a straight line
```bash
python -m timeopt.cli run 2r-path_planning \
  --name twoR_cli \
  --y 0.5 --x0 1.9 --x1 0.5 --n 150 \
  --l1 1.0 --l2 1.0 --tau-max 100 100
```

> If your CLI options differ, use `--help` to see the authoritative signature.

---

## 4) Python One-Liners (no CLI)

### 4.1 CasADi: minimum-time double integrator
```bash
python - <<'PY'
from timeopt.app import MinTimeDoubleIntegrator
prob = MinTimeDoubleIntegrator("di_inline", x0=0.0, xf=1.0, m=1.0, F=10.0)
res = prob.run()
print(res.data)
PY
```

### 4.2 TOPPRA: 2R path time-parameterization (0.6.3 API)
```bash
python - <<'PY'
import numpy as np
from timeopt.design import Planar2RGeom
from timeopt.app import TwoRPathTimeScaler, TwoRParams

geom = Planar2RGeom(l1=1.0, l2=1.0)
qs = geom.path_line_y_const(y=0.5, x0=1.9, x1=0.5, n=150)
prob = TwoRPathTimeScaler("twoR_inline", qs, TwoRParams(tau_max=(100,100)))
res = prob.run()
print(res.data)  # contains tf, grid_size, q0, qf
PY
```

---

## 5) Class Diagram Generation (standalone)

We rely on `py2puml` to generate a PlantUML diagram for the `timeopt` package and then render with `plantuml`.

```bash
# 5.1 Generate .puml
py2puml timeopt time/out/timeopt.puml

# 5.2 Render to PNG (PlantUML must be available; python package `plantuml` or system jar)
plantuml -tpng time/out/timeopt.puml -o .

# Output: time/out/timeopt.png
```

**Notes**
- Ensure Graphviz is installed (PlantUML uses it). On macOS (Homebrew): `brew install graphviz`
- If you prefer the .svg format: `plantuml -tsvg time/out/timeopt.puml -o .`

---

## 6) Building Documentation (Sphinx)

If you keep docs in `docs/`:
```bash
# Create Sphinx skeleton (only once)
sphinx-quickstart docs

# Install theme (Furo already listed above)
pip install furo

# Configure docs/conf.py to include the project path_planning and extensions like autodoc:
#   extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
#   html_theme = "furo"
#   sys.path_planning.insert(0, os.path_planning.abspath(".."))

# Autodoc stubs (example)
sphinx-apidoc -o docs/api timeopt

# Build HTML
sphinx-build -b html docs docs/_build/html

# Or live-reload while editing (if sphinx-autobuild installed)
sphinx-autobuild docs docs/_build/html
```

---

## 7) Troubleshooting

- **Ipopt not installed**: no problem—`MinTimeDoubleIntegrator` falls back to CasADi `sqpmethod` automatically.
- **TOPPRA errors about shapes**: we implemented a shape-stable `inv_dyn` callback (returns `(2,)` for a single sample and `(N,2)` for batches). If you modify it, keep that contract.
- **PlantUML not found**: install via `pip install plantuml` (Python wrapper) or install Java + PlantUML jar. Ensure `graphviz` is installed for PNG/SVG rendering.
- **macOS Gatekeeper** sometimes blocks graphviz binaries; if `dot` is not found, try `brew reinstall graphviz` and ensure your PATH includes `/opt/homebrew/bin`.

---

## 8) Repro Commands (copy‑paste block)

```bash
# Activate env
source .venv/bin/activate

# Run tests
pytest timeopt/tests -q

# Double integrator quick run
python - <<'PY'
from timeopt.app import MinTimeDoubleIntegrator
print(MinTimeDoubleIntegrator("di", 0.0, 1.0, 1.0, 10.0).run().data)
PY

# 2R TOPPRA quick run
python - <<'PY'
from timeopt.design import Planar2RGeom
from timeopt.app import TwoRPathTimeScaler, TwoRParams
geom = Planar2RGeom(1.0, 1.0)
qs = geom.path_line_y_const(y=0.5, x0=1.9, x1=0.5, n=150)
print(TwoRPathTimeScaler("twoR", qs, TwoRParams(tau_max=(100,100))).run().data)
PY

# Class diagram
py2puml timeopt time/out/timeopt.puml
plantuml -tpng time/out/timeopt.puml -o .

# Build docs (if docs/ present)
sphinx-build -b html docs docs/_build/html
```

---

**Happy hacking 🚀**
