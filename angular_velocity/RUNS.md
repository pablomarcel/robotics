# Angular Kinematics — Run Commands (Clean)

This document lists **only** the run-from-project-root Python commands (plus a one-time bootstrap)
you’ll use with the `angular` module. Each command is in its own bash block.

> Conventions
> - Inputs live in `angular/in/`, outputs in `angular/out/`.
> - The primary CLI is `runroot python -m angular.app`.
> - The diagram generator CLI is `runroot python -m angular.tools.diagram`.
> - All commands assume you are anywhere inside the repo; `runroot` will execute from the project root.

---

## -1) One-time session bootstrap (copy/paste once per new shell)

```bash
# --- run-from-root helpers ----------------------------------------------------
_mc_root() {
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse --show-toplevel
    return
  fi
  local d="$PWD"
  while [ "$d" != "/" ]; do
    if [ -d "$d/.git" ] || [ -f "$d/pytest.ini" ] || [ -f "$d/pyproject.toml" ]; then
      echo "$d"; return
    fi
    d="$(dirname "$d")"
  done
  echo "$PWD"
}
runroot() { ( cd "$(_mc_root)" && "$@" ); }
runroot mkdir -p angular_velocity/out
# -----------------------------------------------------------------------------
```

---

## 0) Environment (install/upgrade tools)

```bash
runroot python -m pip install -U pip
```

```bash
runroot python -m pip install numpy pytest pytest-cov pylint graphviz
```

---

## 1) Test Suite (TDD)

```bash
runroot python -m pytest angular_velocity/tests -q
```

```bash
runroot python -m pytest angular_velocity/tests -vv
```

```bash
runroot python -m pytest --cov=angular_velocity --cov-report=term-missing angular_velocity/tests -q
```

---

## 2) Angular CLI — `angular.app`

### Build Rotation from Euler

```bash
runroot python -m angular_velocity.app from-euler --order ZYX --angles 0.3,0.1,-0.2
```

```bash
runroot python -m angular_velocity.app from-euler --order ZYX --angles 0.3,0.1,-0.2 --save R_zyx.npy
```

### Compute ω̃ from Ṙ and R

```bash
runroot python -m angular_velocity.app omega-from-Rdot --R 1,0,0,0,1,0,0,0,1 --Rdot 0,-0.3,0.2,0.3,0,-0.1,-0.2,0.1,0
```

### Compute Ṙ from ω and R

```bash
runroot python -m angular_velocity.app Rdot-from-omega --R 1,0,0,0,1,0,0,0,1 --omega 0.1,0.2,0.3
```

### Velocity Matrix  V = Ṫ T^{-1}

```bash
runroot python -m angular_velocity.app velocity_kinematics-matrix --R 1,0,0,0,1,0,0,0,1 --d 1,2,3 --Rdot 0,-0.3,0.2,0.3,0,-0.1,-0.2,0.1,0 --ddot 0.5,-0.2,0.7
```

### Rigid-Body Point Velocity  v_P = ω×(r_P - d_B) + ḋ_B

```bash
runroot python -m angular_velocity.app rigid-pt-vel --omega 0,0,2 --rP 1,2,0 --dB 1,0,0 --dBdot 0,1,0
```

### Screw Decomposition from Twist

```bash
runroot python -m angular_velocity.app screw-from-twist --twist 0,0,1,0,0.2,0.1
```

---

## 3) Class Diagram Generator — `angular.tools.diagram`

### Emit Graphviz DOT text

```bash
runroot python -m angular_velocity.tools.diagram dot --out angular_velocity/out/classes.dot
```

```bash
runroot python -m angular_velocity.tools.diagram dot --out ""
```

### Emit PlantUML text

```bash
runroot python -m angular_velocity.tools.diagram plantuml --out angular_velocity/out/classes.puml
```

```bash
runroot python -m angular_velocity.tools.diagram plantuml --out ""
```

### Export JSON model

```bash
runroot python -m angular_velocity.tools.diagram json --out angular_velocity/out/classes.json
```

### Render Graphviz image (python-graphviz)

```bash
runroot python -m angular_velocity.tools.diagram graphviz --fmt png --dpi 300 --outstem classes
```

```bash
runroot python -m angular_velocity.tools.diagram graphviz --fmt svg --outstem classes
```

### Render everything (best effort)

```bash
runroot python -m angular_velocity.tools.diagram all
```

### Common option examples

```bash
runroot python -m angular_velocity.tools.diagram dot --rankdir TB --legend --out angular_velocity/out/diagram.dot
```

```bash
runroot python -m angular_velocity.tools.diagram plantuml --theme dark --no-cluster --out angular_velocity/out/diagram.puml
```

```bash
runroot python -m angular_velocity.tools.diagram graphviz --fmt pdf --outstem classes
```

---

## 4) Programmatic UML (pyreverse-style helper)

```bash
runroot python -c "from angular.design import generate_class_diagram; generate_class_diagram()"
```

---

## 5) Quick Sanity Checks

```bash
runroot python -m angular_velocity.app omega-from-Rdot --R 1,0,0,0,1,0,0,0,1 --Rdot 0,-0.3,0.2,0.3,0,-0.1,-0.2,0.1,0
```

```bash
runroot python -m angular_velocity.app velocity_kinematics-matrix --R 1,0,0,0,1,0,0,0,1 --d 1,2,3 --Rdot 0,-0.3,0.2,0.3,0,-0.1,-0.2,0.1,0 --ddot 0.5,-0.2,0.7
```

```bash
runroot python -m angular_velocity.app rigid-pt-vel --omega 0,0,2 --rP 1,2,0 --dB 1,0,0 --dBdot 0,1,0
```

```bash
runroot python -m angular_velocity.app screw-from-twist --twist 0,0,1,0,0.2,0.1
```
