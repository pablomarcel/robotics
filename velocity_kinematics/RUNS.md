# Velocity Kinematics — Run Commands (from repo root)

All commands **must be run from the root of the `robotics` repo** and begin with `runroot python`.
Inputs live in `velocity/in/` and outputs in `velocity/out/`.

> Tip: Create the output directory once: `mkdir -p velocity/out`

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
runroot mkdir -p velocity_kinematics/out
# -----------------------------------------------------------------------------
```

## 0) Environment
```bash
# (Optional) install extras used below
runroot python -m pip install urdfpy cairosvg graphviz
```

## 1) Forward Kinematics (FK)

### 1.1 DH robot (Planar 2R)

# FK at q = [0.3, -0.4]

```bash
runroot python -m velocity_kinematics.app fk velocity_kinematics/in/planar2r.yaml \
  --q 0.3,-0.4 > out/planar2r_fk.json
```

### 1.2 URDF robot (`simple_arm.urdf`)
```bash
# FK at q = [0.3, 0.2]
runroot python -m velocity_kinematics.app fk velocity_kinematics/in/simple_arm.urdf \
  --q 0.3,0.2 > out/urdf_fk.json
```

## 2) Geometric & Analytic Jacobians

### 2.1 Geometric Jacobian (DH)

```bash
runroot python -m velocity_kinematics.app jacobian velocity_kinematics/in/planar2r.yaml \
  --q 0.3,-0.4 > out/planar2r_J.json
```

### 2.2 Analytic Jacobian (DH; Euler ZYX)

```bash
runroot python -m velocity_kinematics.app jacobian-analytic velocity_kinematics/in/planar2r.yaml \
  --q 0.3,-0.4 --euler ZYX > out/planar2r_JA_zyx.json
```

### 2.3 Analytic Jacobian (DH; Euler ZXZ — common for spherical wrists)
```bash
runroot python -m velocity_kinematics.app jacobian-analytic velocity_kinematics/in/spherical6r.yaml \
  --q 0.2,-0.6,0.3,-0.4,0.5,-0.2 \
  --euler ZXZ > out/spherical6r_JA_zxz.json
```

### 2.4 Geometric / Analytic Jacobian (URDF)

# Geometric

```bash
runroot python -m velocity_kinematics.app jacobian velocity_kinematics/in/simple_arm.urdf \
  --q 0.3,0.2 > out/urdf_J.json
```

# Analytic (ZYX)

```bash
runroot python -m velocity_kinematics.app jacobian-analytic velocity_kinematics/in/simple_arm.urdf \
  --q 0.3,0.2 \
  --euler ZYX > out/urdf_JA_zyx.json
```

## 3) Resolved Rates (Inverse Velocity)

### 3.1 Basic (DH)

# Given task-space Xdot = [vx,vy,vz, wx,wy,wz]

```bash
runroot python -m velocity_kinematics.app resolved-rates velocity_kinematics/in/planar2r.yaml \
  --q "0.3,-0.4" \
  --xdot "0.05,0,0,0,0,0" \
  --out velocity_kinematics/out/planar2r_qdot.json
```

### 3.2 With damping & task weights

```bash
runroot python -m velocity_kinematics.app resolved-rates velocity_kinematics/in/planar2r.yaml \
  --q 0.3,-0.4 \
  --xdot 0.05,0,0,0,0,0 \
  --damping 1e-3 \
  --weights 1,1,1,0.2,0.2,0.2 > out/planar2r_qdot_weighted.json
```

### 3.3 URDF
```bash
runroot python -m velocity_kinematics.app resolved-rates velocity_kinematics/in/simple_arm.urdf \
  --q 0.3,0.2 \
  --xdot 0.0,0.02,0.0,0,0,0 > out/urdf_qdot.json
```

## 4) Newton–Raphson Pose IK

### 4.1 Position-only target (DH)

```bash
runroot python -m velocity_kinematics.app newton-ik velocity_kinematics/in/planar2r.yaml \
  --q0 0.1,0.1 \
  --p 1.6,0.1,0.0 \
  --max-iter 100 > out/planar2r_ik_pos.json
```

### 4.2 Position + Orientation (Euler ZYX)

# Target Euler angles in degrees (default). Use --rad to pass radians.

```bash
runroot python -m velocity_kinematics.app newton-ik velocity_kinematics/in/spherical6r.yaml \
  --q0 0,0,0,0,0,0 \
  --p 0.3,0.1,0.2 \
  --euler ZYX \
  --angles 10,20,0 \
  --max-iter 60 > out/spherical6r_ik_zyx.json
```

### 4.3 Orientation-only (Euler ZXZ) on wrist

```bash
runroot python -m velocity_kinematics.app newton-ik velocity_kinematics/in/spherical6r.yaml \
  --q0 0,0,0,0,0,0 \
  --euler ZXZ \
  --angles 15,30,45 \
  --max-iter 60 > out/spherical6r_ik_zxz_only.json
```

### 4.4 IK with task weights

```bash
runroot python -m velocity_kinematics.app newton-ik velocity_kinematics/in/spherical6r.yaml \
  --q0 0,0,0,0,0,0 \
  --p 0.3,0.2,0.1 \
  --euler ZYX \
  --angles 0,0,0 \
  --weights 1,1,1,0.5,0.5,0.5 > out/spherical6r_ik_weighted.json
```

## 5) Linear Algebra Utilities

# Solve A x = b

```bash
runroot python -m velocity_kinematics.app lu-solve \
  --A "[[2,1],[1,3]]" \
  --b "[1,2]" \
  --out velocity_kinematics/out/lu_solve.json
```

# Inverse via LU

```bash
runroot python -m velocity_kinematics.app lu-inv \
  --A velocity_kinematics/in/matrix.json > out/lu_inv.json
```

## 6) Codebase Class Diagrams (UML)

# DOT

```bash
runroot python -m velocity_kinematics.tools.diagram dot \
  --out classes.dot
```

# PlantUML

```bash
runroot python -m velocity_kinematics.tools.diagram plantuml \
  --out classes.puml
```

# Graphviz render (PNG/SVG/PDF) — requires `graphviz` runroot python package; no system binary needed

```bash
runroot python -m velocity_kinematics.tools.diagram graphviz \
  --fmt svg \
  --outstem classes
```

# Everything at once (JSON model, DOT, PUML, and best-effort Graphviz)

```bash
runroot python -m velocity_kinematics.tools.diagram all
```

## 7) Robot Renders (DH specs → SVG/PNG)

# Simple SVG (defaults q=zeros)

```bash
runroot python -m velocity_kinematics.tools.robotviz draw velocity_kinematics/in/planar2r.yaml \
  --out velocity_kinematics/out/planar2r.svg
```

# With configuration and dark theme

```bash
runroot python -m velocity_kinematics.tools.robotviz draw velocity_kinematics/in/planar2r.yaml \
  --q 0.3,-0.4 \
  --theme dark \
  --out velocity_kinematics/out/planar2r_dark.svg
```

# PNG output (uses cairosvg under the hood)

```bash
runroot python -m velocity_kinematics.tools.robotviz draw velocity_kinematics/in/planar2r.yaml \
  --out velocity_kinematics/out/planar2r.png
```

# XZ projection (view along Y)

```bash
runroot python -m velocity_kinematics.tools.robotviz draw velocity_kinematics/in/planar2r.yaml \
  --view xz \
  --out velocity_kinematics/out/planar2r_xz.svg
```

# Isometric projection

```bash
runroot python -m velocity_kinematics.tools.robotviz draw velocity_kinematics/in/planar2r.yaml \
  --view iso \
  --out velocity_kinematics/out/planar2r_iso.svg
```

> Note: `robotviz` currently renders **DH** models. If you want `.urdf` renders here, we can extend it.

## 8) File Cheat Sheet (inputs/outputs)

- Inputs: `velocity/in/planar2r.yaml`, `velocity/in/spherical6r.yaml`, `velocity/in/simple_arm.urdf`, plus any JSON arrays/matrices you add.
- Outputs: All examples write into `velocity/out/*.json` or `velocity/out/*.svg|png`.

## 9) Troubleshooting

- **URDF errors** → Ensure `urdfpy` imports cleanly with `runroot python -c "import urdfpy; print('ok')"`.
- **Analytic Jacobian near singularities** → The tool gracefully falls back to geometric Jacobian.
- **Angles unit** → `newton-ik --angles` defaults to **degrees**; add `--rad` to interpret as radians.
- **Run directory** → Always run from repo root so relative `velocity/in/...` paths resolve.

Happy hacking 🚀

### Sphinx

python -m velocity_kinematics.cli sphinx-skel velocity_kinematics/docs
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html
sphinx-autobuild docs docs/_build/html