# Velocity Kinematics — Run Commands (from repo root)

All commands **must be run from the root of the `robotics` repo** and begin with `python`.
Inputs live in `velocity/in/` and outputs in `velocity/out/`.

> Tip: Create the output directory once: `mkdir -p velocity/out`

---

## 0) Environment
```bash
# (Optional) install extras used below
python -m pip install urdfpy cairosvg graphviz
```

---

## 1) Forward Kinematics (FK)

### 1.1 DH robot (Planar 2R)
```bash
# FK at q = [0.3, -0.4]
python -m velocity.app fk velocity/in/planar2r.yaml --q 0.3,-0.4 > velocity/out/planar2r_fk.json
```

### 1.2 URDF robot (`simple_arm.urdf`)
```bash
# FK at q = [0.3, 0.2]
python -m velocity.app fk velocity/in/simple_arm.urdf --q 0.3,0.2 > velocity/out/urdf_fk.json
```

---

## 2) Geometric & Analytic Jacobians

### 2.1 Geometric Jacobian (DH)
```bash
python -m velocity.app jacobian velocity/in/planar2r.yaml --q 0.3,-0.4 > velocity/out/planar2r_J.json
```

### 2.2 Analytic Jacobian (DH; Euler ZYX)
```bash
python -m velocity.app jacobian-analytic velocity/in/planar2r.yaml --q 0.3,-0.4 --euler ZYX > velocity/out/planar2r_JA_zyx.json
```

### 2.3 Analytic Jacobian (DH; Euler ZXZ — common for spherical wrists)
```bash
python -m velocity.app jacobian-analytic velocity/in/spherical6r.yaml --q 0.2,-0.6,0.3,-0.4,0.5,-0.2 --euler ZXZ > velocity/out/spherical6r_JA_zxz.json
```

### 2.4 Geometric / Analytic Jacobian (URDF)
```bash
# Geometric
python -m velocity.app jacobian velocity/in/simple_arm.urdf --q 0.3,0.2 > velocity/out/urdf_J.json
# Analytic (ZYX)
python -m velocity.app jacobian-analytic velocity/in/simple_arm.urdf --q 0.3,0.2 --euler ZYX > velocity/out/urdf_JA_zyx.json
```

---

## 3) Resolved Rates (Inverse Velocity)

### 3.1 Basic (DH)
```bash
# Given task-space Xdot = [vx,vy,vz, wx,wy,wz]
python -m velocity.app resolved-rates velocity/in/planar2r.yaml --q 0.3,-0.4 --xdot 0.05,0.00,0.00, 0,0,0 > velocity/out/planar2r_qdot.json
```

### 3.2 With damping & task weights
```bash
python -m velocity.app resolved-rates velocity/in/planar2r.yaml --q 0.3,-0.4 --xdot 0.05,0,0, 0,0,0 --damping 1e-3 --weights 1,1,1, 0.2,0.2,0.2 > velocity/out/planar2r_qdot_weighted.json
```

### 3.3 URDF
```bash
python -m velocity.app resolved-rates velocity/in/simple_arm.urdf --q 0.3,0.2 --xdot 0.0,0.02,0.0, 0,0,0 > velocity/out/urdf_qdot.json
```

---

## 4) Newton–Raphson Pose IK

### 4.1 Position-only target (DH)
```bash
python -m velocity.app newton-ik velocity/in/planar2r.yaml --q0 0.1,0.1 --p 1.6,0.1,0.0 --max-iter 100 > velocity/out/planar2r_ik_pos.json
```

### 4.2 Position + Orientation (Euler ZYX)
```bash
# Target Euler angles in degrees (default). Use --rad to pass radians.
python -m velocity.app newton-ik velocity/in/spherical6r.yaml --q0 0,0,0,0,0,0 --p 0.3,0.1,0.2 --euler ZYX --angles 10,20,0 --max-iter 60 > velocity/out/spherical6r_ik_zyx.json
```

### 4.3 Orientation-only (Euler ZXZ) on wrist
```bash
python -m velocity.app newton-ik velocity/in/spherical6r.yaml --q0 0,0,0,0,0,0 --euler ZXZ --angles 15,30,45 --max-iter 60 > velocity/out/spherical6r_ik_zxz_only.json
```

### 4.4 IK with task weights
```bash
python -m velocity.app newton-ik velocity/in/spherical6r.yaml --q0 0,0,0,0,0,0 --p 0.3,0.2,0.1 --euler ZYX --angles 0,0,0 --weights 1,1,1, 0.5,0.5,0.5 > velocity/out/spherical6r_ik_weighted.json
```

---

## 5) Linear Algebra Utilities

```bash
# Solve A x = b
python -m velocity.app lu-solve --A "[[2,1],[1,3]]" --b "[1,2]" > velocity/out/lu_solve.json

# Inverse via LU
python -m velocity.app lu-inv --A velocity/in/matrix.json > velocity/out/lu_inv.json
```

---

## 6) Codebase Class Diagrams (UML)

```bash
# DOT
python -m velocity.tools.diagram dot --out velocity/out/classes.dot

# PlantUML
python -m velocity.tools.diagram plantuml --out velocity/out/classes.puml

# Graphviz render (PNG/SVG/PDF) — requires `graphviz` Python package; no system binary needed
python -m velocity.tools.diagram graphviz --fmt svg --outstem classes
# -> writes velocity/out/classes.svg

# Everything at once (JSON model, DOT, PUML, and best-effort Graphviz)
python -m velocity.tools.diagram all
```

---

## 7) Robot Renders (DH specs → SVG/PNG)

```bash
# Simple SVG (defaults q=zeros)
python -m velocity.tools.robotviz draw velocity/in/planar2r.yaml --out velocity/out/planar2r.svg

# With configuration and dark theme
python -m velocity.tools.robotviz draw velocity/in/planar2r.yaml --q 0.3,-0.4 --theme dark --out velocity/out/planar2r_dark.svg

# PNG output (uses cairosvg under the hood)
python -m velocity.tools.robotviz draw velocity/in/planar2r.yaml --out velocity/out/planar2r.png

# XZ projection (view along Y)
python -m velocity.tools.robotviz draw velocity/in/planar2r.yaml --view xz --out velocity/out/planar2r_xz.svg

# Isometric projection
python -m velocity.tools.robotviz draw velocity/in/planar2r.yaml --view iso --out velocity/out/planar2r_iso.svg
```

> Note: `robotviz` currently renders **DH** models. If you want `.urdf` renders here, we can extend it.

---

## 8) File Cheat Sheet (inputs/outputs)

- Inputs: `velocity/in/planar2r.yaml`, `velocity/in/spherical6r.yaml`, `velocity/in/simple_arm.urdf`, plus any JSON arrays/matrices you add.
- Outputs: All examples write into `velocity/out/*.json` or `velocity/out/*.svg|png`.

---

## 9) Troubleshooting

- **URDF errors** → Ensure `urdfpy` imports cleanly with `python -c "import urdfpy; print('ok')"`.
- **Analytic Jacobian near singularities** → The tool gracefully falls back to geometric Jacobian.
- **Angles unit** → `newton-ik --angles` defaults to **degrees**; add `--rad` to interpret as radians.
- **Run directory** → Always run from repo root so relative `velocity/in/...` paths resolve.

Happy hacking 🚀