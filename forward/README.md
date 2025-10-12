
# Forward Robotics – Run Commands Cheat Sheet

This README collects **all** common/edge commands for the `forward` module: validation, schemas,
FK, Jacobians, presets, diagram generation (DOT/PlantUML/Graphviz), and docs (Sphinx).

> Conventions used below:
>
> - Input specs live in `forward/in/` and outputs in `forward/out/` (customize as you like).
> - All commands assume you’re in your project root and your venv is active.
> - Replace paths as needed. Use `--help` on any command to see options.

---

## 0) Quick sanity

```bash
python -m forward.cli --help
python -m forward.cli fk --help
python -m forward.cli jacobian-space --help
python -m forward.cli jacobian-body --help
python -m forward.cli preset-scara --help
python -m forward.cli preset-wrist --help
python -m forward.cli diagram-dot --help
python -m forward.cli sphinx-skel --help
```

---

## 1) Validate robot specs + export schema

### Validate JSON / YAML spec
```bash
# JSON
python -m forward.cli validate forward/in/my_robot.json

# YAML
python -m forward.cli validate forward/in/my_robot.yaml
```

### Export JSON Schema for specs
```bash
python -m forward.cli schema -o forward/out/robot.schema.json
```

---

## 2) Forward Kinematics (FK)

### JSON spec → JSON/CSV/NPY transform
```bash
# JSON output
python -m forward.cli fk forward/in/my_robot.json --q 0.0 --q 1.0 --q 0.2 -o forward/out/T.json

# CSV output
python -m forward.cli fk forward/in/my_robot.json --q 1.57 --q 0.0 --q 0.1 -o forward/out/T.csv

# NPY (NumPy) output
python -m forward.cli fk forward/in/my_robot.yaml --q 0.3 --q -0.2 --q 0.7 -o forward/out/T.npy
```

**Notes**
- Repeat `--q` for each joint value (order is joint index order).
- Output format is inferred from the `-o/--out` extension: `.json`, `.csv`, `.npy`.

---

## 3) Jacobians (space / body)

### Space Jacobian
```bash
python -m forward.cli jacobian-space forward/in/my_robot.json --q 0.1 --q 0.2 --q 0.3 -o forward/out/J_space.json
python -m forward.cli jacobian-space forward/in/my_robot.yaml --q 0.0 --q 0.0 --q 0.0 -o forward/out/J_space.npy
```

### Body Jacobian
```bash
python -m forward.cli jacobian-body forward/in/my_robot.json --q 0.1 --q 0.2 --q 0.3 -o forward/out/J_body.json
python -m forward.cli jacobian-body forward/in/my_robot.yaml --q 0.0 --q 0.0 --q 0.0 -o forward/out/J_body.csv
```

**Analytical Jacobians**
- The library also supports analytical Jacobian derivations; use the same commands above.
  (The adjoint relations are implemented; analytical forms are exposed where applicable.)

---

## 4) Presets

### SCARA
```bash
python -m forward.cli preset-scara   --l1 0.7 --l2 0.6 --d 0.18   --q 0.0 --q 1.2 --q 0.05   -o forward/out/scara_T.json

# Expect companion file: forward/out/scara_T_J_space.json
```

### Spherical wrist (types 1–3)
```bash
# type 1 (with d7)
python -m forward.cli preset-wrist --type 1 --d7 0.12 --q 0.1 --q 0.2 --q 0.3 -o forward/out/wrist_T.json
# creates forward/out/wrist_T_J_body.json

# type 2 (no d7)
python -m forward.cli preset-wrist --type 2 --q 0.2 --q -0.4 --q 1.0 -o forward/out/wrist2_T.json
# creates forward/out/wrist2_T_J_body.json

# type 3 (no d7)
python -m forward.cli preset-wrist --type 3 --q 0.0 --q 0.0 --q 0.0 -o forward/out/wrist3_T.json
# creates forward/out/wrist3_T_J_body.json
```

---

## 5) I/O formats and helpers

- **Spec formats**: `.json` and `.yaml` / `.yml` are supported.  
- **Outputs**: transforms and Jacobians support `.json`, `.csv`, `.npy` by filename.

Tips:
```bash
# Quick peek
cat forward/out/T.json
python -c "import numpy as np; print(np.load('forward/out/J_space.npy'))"
```

---

## 6) Diagrams (class diagrams)

### DOT (Graphviz text)
```bash
# Write DOT to file
python -m forward.cli diagram-dot -o forward/out/classes.dot
```

### PlantUML (.puml text)
```bash
# Using the tool module directly; writes to forward/out/classes.puml
python -m forward.tools.diagram plantuml --out forward/out/classes.puml
# Upload classes.puml to any PlantUML online renderer to get PNG/SVG.
```

### Optional: Render PNG without the `dot` binary (if python-graphviz installed)
```bash
pip install graphviz   # the Python package only
python -m forward.tools.diagram graphviz --fmt png --dpi 300 --outstem classes
# -> forward/out/classes.png
```

**Styling options (both dot/plantuml):**
```bash
# Dark theme + legend
python -m forward.tools.diagram dot --theme dark --legend --out forward/out/classes.dot
python -m forward.tools.diagram plantuml --theme dark --legend --out forward/out/classes.puml

# No clustering by module
python -m forward.tools.diagram dot --no-cluster --out forward/out/classes.dot
```

---

## 7) Sphinx docs

> No Xcode/CLT required — build via Python.

### Generate skeleton
```bash
python -m forward.cli sphinx-skel docs
# Options:
#   --project "My Robotics FK" --package forward --author "Your Name" --force
```

### Build HTML (no `make`)
```bash
# from project root
pip install sphinx furo
python -m sphinx -b html docs docs/_build/html
# open docs/_build/html/index.html
```

### Live rebuild (optional)
```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
# http://127.0.0.1:8000/
```

### (Optional) API pages per module
```bash
sphinx-apidoc -o docs/api forward
python -m sphinx -b html docs docs/_build/html
```

---

## 8) Testing & coverage

```bash
pytest forward/tests -q

# With coverage (if configured in pytest.ini or add -q -vv as you prefer)
pytest --maxfail=1 --disable-warnings -q
```

---

## 9) End-to-end examples

```bash
# Validate, run FK and Jacobians, then diagram and docs
python -m forward.cli validate forward/in/my_robot.json
python -m forward.cli fk forward/in/my_robot.json --q 0.0 --q 0.0 --q 0.0 -o forward/out/T.json
python -m forward.cli jacobian-space forward/in/my_robot.json --q 0.0 --q 0.0 --q 0.0 -o forward/out/J_space.json
python -m forward.cli jacobian-body  forward/in/my_robot.json --q 0.0 --q 0.0 --q 0.0 -o forward/out/J_body.json

python -m forward.cli preset-scara --l1 0.7 --l2 0.6 --d 0.18 --q 0.0 --q 1.2 --q 0.05 -o forward/out/scara_T.json
python -m forward.cli preset-wrist --type 2 --q 0.2 --q -0.4 --q 1.0 -o forward/out/wrist2_T.json

python -m forward.cli diagram-dot -o forward/out/classes.dot
python -m forward.tools.diagram plantuml --out forward/out/classes.puml

python -m forward.cli sphinx-skel docs
python -m sphinx -b html docs docs/_build/html
```

---

## 10) Troubleshooting

- **“q has length N, expected M”** – supply the correct number of `--q` repeats for the chain.
- **YAML not found** – install PyYAML: `pip install pyyaml`.
- **Can’t render PNG via `dot`** – use `python -m forward.tools.diagram plantuml --out forward/out/classes.puml` and render online,
  or install the Python `graphviz` package and use the `graphviz` subcommand (no system binary needed).
- **Sphinx warnings** – they’re usually docstring formatting nits; HTML still builds.
