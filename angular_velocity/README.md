
# Angular Kinematics — Run Commands (Full & Exhaustive)

This README collects **all run commands** you’ll use for the `angular` module
(OOP, TDD-first toolkit for angular velocity, Eq. **7.1–7.416**).

> Conventions
> - Inputs live in `angular/in/`, outputs in `angular/out/`.
> - The CLI is exposed via `python -m angular.app`.
> - The diagram generator CLI is exposed via `python -m angular.tools.diagram`.
> - All examples assume you’re in the project root (where the `angular/` package lives).

---

## 0) Environment & Installation

```bash
# create / activate a virtual environment (example: venv)
python -m venv .venv
source .venv/bin/activate            # (Linux/macOS)
# or: .venv\Scripts\activate       # (Windows PowerShell)

# install runtime & dev deps (adjust as needed)
pip install -U pip
pip install numpy graphviz pylint pytest pytest-cov

# if you have a pyproject/setup, you can install the package itself (editable)
# pip install -e .
```

Optional extras for diagrams:
```bash
# Needed to render .dot -> images via python-graphviz
pip install graphviz

# (Optional) For pyreverse (alternative UML using pylint)
pip install pylint
```

---

## 1) Run the Test Suite (TDD)

```bash
# quick
pytest angular_velocity/tests -q

# verbose
pytest angular_velocity/tests -vv

# with coverage (example)
pytest --cov=angular_velocity --cov-report=term-missing angular_velocity/tests -q
```

---

## 2) Angular CLI — `python -m angular.app`

### 2.1 Build Rotation from Euler

```bash
python -m angular_velocity.app from-euler --order ZYX --angles 0.3,0.1,-0.2
# Save to angular_velocity/out/
python -m angular_velocity.app from-euler --order ZYX --angles 0.3,0.1,-0.2 --save R_zyx.npy
```

**Flags**
- `--order` : Euler order (e.g., `ZYX`, `XYZ`, `Z`, etc.).
- `--angles`: `phi,theta,psi` as **comma-separated** floats (3 values).

### 2.2 Compute ω̃ from Ṙ and R

```bash
python -m angular_velocity.app omega-from-Rdot   --R 1,0,0,0,1,0,0,0,1   --Rdot 0,-0.3,0.2,0.3,0,-0.1,-0.2,0.1,0
```

**Flags**
- `--R`    : 3×3 rotation (9 comma-separated floats, row-major).
- `--Rdot` : 3×3 time derivative of `R` (9 comma-separated floats).

### 2.3 Compute Ṙ from ω and R

```bash
python -m angular_velocity.app Rdot-from-omega   --R 1,0,0,0,1,0,0,0,1   --omega 0.1,0.2,0.3
```

**Flags**
- `--R`     : 3×3 rotation.
- `--omega` : 3-vector (rad/s).

### 2.4 Velocity Matrix \( V = \. T \, T^{-1} \)

```bash
python -m angular_velocity.app velocity_kinematics-matrix   --R 1,0,0,0,1,0,0,0,1   --d 1,2,3   --Rdot 0,-0.3,0.2,0.3,0,-0.1,-0.2,0.1,0   --ddot 0.5,-0.2,0.7
```

**Flags**
- `--R`    : 3×3 rotation.
- `--d`    : 3-vector origin position.
- `--Rdot` : 3×3 rotation derivative.
- `--ddot` : 3-vector origin linear velocity.

### 2.5 Rigid-Body Point Velocity \( v_P = ω×(r_P - d_B) + \. d_B \)

```bash
python -m angular_velocity.app rigid-pt-vel   --omega 0,0,2   --rP 1,2,0   --dB 1,0,0   --dBdot 0,1,0
```

### 2.6 Screw Decomposition from Twist

```bash
python -m angular_velocity.app screw-from-twist --twist 0,0,1,0,0.2,0.1
# prints JSON payload with axis s, moment m, and pitch p
```

---

## 3) Class Diagram Generator — `python -m angular.tools.diagram`

All outputs default to `angular/out/`. You can customize `--outdir`.

### 3.1 Emit Graphviz DOT Text

```bash
python -m angular_velocity.tools.diagram dot --out angular_velocity/out/classes.dot
# print to stdout
python -m angular_velocity.tools.diagram dot --out ""
```

### 3.2 Emit PlantUML Text

```bash
python -m angular_velocity.tools.diagram plantuml --out angular_velocity/out/classes.puml
# print to stdout
python -m angular_velocity.tools.diagram plantuml --out ""
```

### 3.3 Export JSON Model

```bash
python -m angular_velocity.tools.diagram json --out angular_velocity/out/classes.json
```

### 3.4 Render Graphviz (PNG/SVG/PDF) via python-graphviz

```bash
# PNG with DPI
python -m angular_velocity.tools.diagram graphviz --fmt png --dpi 300 --outstem classes

# SVG (resolution independent; ignores --dpi)
python -m angular_velocity.tools.diagram graphviz --fmt svg --outstem classes
```

### 3.5 Render Everything (best effort)

```bash
python -m angular_velocity.tools.diagram all
# Produces: JSON, DOT, PlantUML, and tries a Graphviz image.
```

**Common options (apply to all subcommands):**

- `--packages`: Comma-separated roots. Default:
  ```
  angular.core,angular.io,angular.design,angular.utils,angular.app,angular.apis,angular.cli
  ```
- `--outdir`: Output directory (default: `angular/out`).
- `--theme`: `light` | `dark`.
- `--rankdir`: `LR` | `TB` (left-right or top-bottom).
- `--legend`: include a legend box.
- `--no-cluster`: disable package clustering.

Examples:
```bash
python -m angular_velocity.tools.diagram dot --rankdir TB --legend --out angular_velocity/out/diagram.dot
python -m angular_velocity.tools.diagram plantuml --theme dark --no-cluster --out angular_velocity/out/diagram.puml
python -m angular_velocity.tools.diagram graphviz --fmt pdf --outstem classes
```

---

## 4) Programmatic Class Diagram (pyreverse alternative)

If you prefer a one-shot function (requires `pylint` and Graphviz installed system-wide):

```bash
python -c "from angular.design import generate_class_diagram; generate_class_diagram()"
# Outputs PNGs into angular_velocity/out/ (if the helper exists in your tree)
```

> Note: The preferred way for this repo is the CLI in `angular.tools.diagram` (section 3).

---

## 5) Input/Output File Conventions

- Place any raw inputs in: `angular/in/`
- Expect artifacts to land in: `angular/out/` (e.g., `.npy`, `.dot`, `.puml`, `.png`)

Example:
```bash
python -m angular_velocity.app from-euler --order ZYX --angles 0.3,0.1,-0.2 --save R_zyx.npy
ls angular_velocity/out/
# R_zyx.npy
```

---

## 6) Troubleshooting

- **NumPy errors**: Ensure arrays are comma-separated without spaces or quote the whole argument.
- **graphviz render errors**: Install the `graphviz` Python package. If rendering still fails, ensure the Graphviz binaries are installed on your system or stick with DOT/PUML text outputs.
- **Import issues**: Confirm you’re running from the project root and the `angular/` package is on `PYTHONPATH` (or install with `pip install -e .`).

---

## 7) Quick Sanity Checks

```bash
# 1) ω̃ from Ṙ Rᵀ
python -m angular_velocity.app omega-from-Rdot --R 1,0,0,0,1,0,0,0,1 --Rdot 0,-0.3,0.2,0.3,0,-0.1,-0.2,0.1,0

# 2) V = Ṫ T^{-1}
python -m angular_velocity.app velocity_kinematics-matrix --R 1,0,0,0,1,0,0,0,1 --d 1,2,3   --Rdot 0,-0.3,0.2,0.3,0,-0.1,-0.2,0.1,0 --ddot 0.5,-0.2,0.7

# 3) Rigid-body point velocity_kinematics
python -m angular_velocity.app rigid-pt-vel --omega 0,0,2 --rP 1,2,0 --dB 1,0,0 --dBdot 0,1,0

# 4) Screw decomposition
python -m angular_velocity.app screw-from-twist --twist 0,0,1,0,0.2,0.1
```
