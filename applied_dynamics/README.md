# Applied Dynamics — Run Commands Cheat Sheet

This README collects **all the practical run commands** for the `applied` package: CLI presets, diagram tooling, and a few programmatic snippets you can copy/paste.

> Tip: All CLI examples below can be run either as `python -m applied.cli ...` **or** using the module’s entry-point in the same way from your project root. Replace paths as needed.

---

## 0) Environment setup (suggested)

```bash
# 1) Create & activate a venv (if you haven't)
python3 -m venv .venv
source .venv/bin/activate         # on Windows: .venv\Scripts\activate

# 2) Install the project (editable is handy during dev)
pip install -e .

# 3) (Optional) extra tools for diagram rendering
pip install graphviz              # for 'diagram graphviz' command (renders PNG/SVG/PDF)
# Make sure Graphviz binaries are installed on your OS if you want high-quality layouts.
```

---

## 1) CLI — Design presets

List and instantiate ready‑made dynamic models (symbolic and numeric variants).

### 1.1 List available presets
```bash
python -m applied_dynamics.cli design --list
```

### 1.2 Create a preset and print a short summary
```bash
# symbolic
python -m applied_dynamics.cli design --preset pendulum_sym
python -m applied_dynamics.cli design --preset spherical_sym
python -m applied_dynamics.cli design --preset planar2r_sym
python -m applied_dynamics.cli design --preset absorber_sym

# numeric
python -m applied_dynamics.cli design --preset pendulum_num
python -m applied_dynamics.cli design --preset planar2r_num
python -m applied_dynamics.cli design --preset absorber_num
```

### 1.3 Export a small JSON summary
```bash
python -m applied_dynamics.cli design --preset planar2r_num --export applied_dynamics/out/planar2r_num.json
```

> The JSON includes the preset name, model class, and parameter fields.

---

## 2) CLI — Diagram tooling

Generate DOT/PlantUML/JSON metadata for classes in the `applied` package. All subcommands accept **common options**:

- `--packages`   : CSV of packages to scan (default: `applied.core,applied.dynamics,applied.models,applied.io,applied.utils,applied.app,applied.apis,applied.tools`)
- `--outdir`     : Output directory (default: `applied/out`)
- `--theme`      : `light` or `dark` (default: `light`)
- `--rankdir`    : `LR` or `TB` for left‑right or top‑bottom layout (default: `LR`)
- `--legend`     : include a legend block
- `--no-cluster` : do not group by module

### 2.1 Emit Graphviz DOT text
```bash
python -m applied_dynamics.cli diagram dot --out applied_dynamics/out/classes.dot
# with options
python -m applied_dynamics.cli diagram dot   --packages applied_dynamics.core,applied_dynamics.models   --outdir applied_dynamics/out   --theme dark   --rankdir TB   --legend   --no-cluster   --out applied_dynamics/out/classes_custom.dot
```

### 2.2 Emit PlantUML text
```bash
python -m applied_dynamics.cli diagram plantuml --out applied_dynamics/out/classes.puml
```

### 2.3 Export discovered model JSON (structure of classes/relations)
```bash
python -m applied_dynamics.cli diagram json --out applied_dynamics/out/classes.json
```

### 2.4 Render via python‑graphviz (PNG/SVG/PDF)
```bash
# PNG with default DPI
python -m applied_dynamics.cli diagram graphviz --fmt png --dpi 220 --outstem applied_dynamics/out/classes

# SVG
python -m applied_dynamics.cli diagram graphviz --fmt svg --outstem applied_dynamics/out/classes_svg

# PDF, top‑bottom layout and dark theme
python -m applied_dynamics.cli diagram graphviz   --fmt pdf   --dpi 260   --rankdir TB   --theme dark   --outstem applied_dynamics/out/classes_pdf
```

### 2.5 “All” at once (emit JSON, DOT, PlantUML and try Graphviz render)
```bash
python -m applied_dynamics.cli diagram all
# You can still customize the common options, e.g.:
python -m applied_dynamics.cli diagram all   --packages applied_dynamics.core,applied_dynamics.models   --outdir applied_dynamics/out   --theme dark   --rankdir TB   --legend
```

---

## 3) CLI — Preset aliases (smoke‑test style)

For convenience and parity with tests, the CLI accepts **top‑level aliases** that behave like a quick summary of the corresponding preset (exit code 0).

```bash
python -m applied_dynamics.cli pendulum
python -m applied_dynamics.cli spherical
python -m applied_dynamics.cli planar2r
python -m applied_dynamics.cli absorber
```

> These print a one‑screen summary of the model (class name, its parameters, and its q/qd).

---

## 4) Programmatic usage (quick snippets)

### 4.1 Derive pendulum EOM symbolically

```python
from applied_dynamics.apis import AppliedDynamicsAPI

r = AppliedDynamicsAPI().derive_simple_pendulum()
eom, K, V, M = r.data["EOM"], r.data["K"], r.data["V"], r.data["M"]
print(eom), print(K), print(V), print(M)
```

### 4.2 Numeric integration of the pendulum (RK4 fallback if SciPy missing)

```python
from applied_dynamics.design import DesignLibrary
from applied_dynamics.integrators import LagrangeRHS, ODESolver, IntegratorConfig

sys = DesignLibrary().create("pendulum_num")
rhs = LagrangeRHS.from_model(sys)
y0 = rhs.pack_state([0.1], [0.0])  # initial angle 0.1 rad, zero speed
cfg = IntegratorConfig(t_span=(0.0, 1.0), rk4_dt=1e-3)  # SciPy used if installed
sol = ODESolver(cfg).solve(rhs, y0)

print(sol.t.shape, sol.y.shape)  # times and stacked [q; qd]
```

---

## 5) Testing

```bash
pytest -q
# or only the applied_dynamics tests
pytest applied_dynamics/tests -q
```

---

## 6) Outputs

- CLI diagrams and exports default to `applied/out/`
- Trajectory CSVs from programmatic runs (if you call `Trajectory.to_csv`) default to `applied/out/trajectory.csv`

---

## 7) Troubleshooting

- **Graphviz render errors**: ensure the Graphviz system binaries are installed and on your PATH (in addition to the `graphviz` Python package).
- **SymPy simplify mismatches in custom scripts**: avoid mixing the same‐named symbols with different assumptions (e.g., `sp.symbols("m", positive=True)` vs `sp.symbols("m")`).

Happy tinkering! 🛠️
