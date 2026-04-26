
# RUNS.md — Forward Robotics (Run-from-Root Commands)

This document lists **only** the run-from-root Python commands required to use the
`forward` module: validation, schema, FK, Jacobians, presets, diagrams, docs, and tests.

---

## -1) One-time shell bootstrap (paste once per new shell)

```bash
# Find project root: prefer Git; otherwise walk up to a marker file.
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
# Run a command from the project root (without changing your current dir)
runroot() { ( cd "$(_mc_root)" && "$@" ); }
# Ensure default output dir exists
runroot python -c "import pathlib; pathlib.Path('forward/out').mkdir(parents=True, exist_ok=True)"
```

---

## 0) Quick sanity

```bash
runroot python -m forward_kinematics.cli --help
```

```bash
runroot python -m forward_kinematics.cli fk --help
```

```bash
runroot python -m forward_kinematics.cli jacobian-space --help
```

```bash
runroot python -m forward_kinematics.cli jacobian-body --help
```

```bash
runroot python -m forward_kinematics.cli preset-scara --help
```

```bash
runroot python -m forward_kinematics.cli preset-wrist --help
```

```bash
runroot python -m forward_kinematics.cli diagram-dot --help
```

```bash
runroot python -m forward_kinematics.cli sphinx-skel --help
```

---

## 1) Validate specs & export schema

```bash
runroot python -m forward_kinematics.cli validate forward_kinematics/in/my_robot.json
```

```bash
runroot python -m forward_kinematics.cli validate forward_kinematics/in/my_robot.yaml
```

```bash
runroot python -m forward_kinematics.cli schema -o forward_kinematics/out/robot.schema.json
```

---

## 2) Forward kinematics (FK)

```bash
runroot python -m forward_kinematics.cli fk forward_kinematics/in/my_robot.json --q 0.0 --q 1.0 --q 0.2 -o forward_kinematics/out/T.json
```

```bash
runroot python -m forward_kinematics.cli fk forward_kinematics/in/my_robot.json --q 1.57 --q 0.0 --q 0.1 -o forward_kinematics/out/T.csv
```

```bash
runroot python -m forward_kinematics.cli fk forward_kinematics/in/my_robot.yaml --q 0.3 --q -0.2 --q 0.7 -o forward_kinematics/out/T.npy
```

---

## 3) Jacobians

```bash
runroot python -m forward_kinematics.cli jacobian-space forward_kinematics/in/my_robot.json --q 0.1 --q 0.2 --q 0.3 -o forward_kinematics/out/J_space.json
```

```bash
runroot python -m forward_kinematics.cli jacobian-space forward_kinematics/in/my_robot.yaml --q 0.0 --q 0.0 --q 0.0 -o forward_kinematics/out/J_space.npy
```

```bash
runroot python -m forward_kinematics.cli jacobian-body forward_kinematics/in/my_robot.json --q 0.1 --q 0.2 --q 0.3 -o forward_kinematics/out/J_body.json
```

```bash
runroot python -m forward_kinematics.cli jacobian-body forward_kinematics/in/my_robot.yaml --q 0.0 --q 0.0 --q 0.0 -o forward_kinematics/out/J_body.csv
```

---

## 4) Presets

```bash
runroot python -m forward_kinematics.cli preset-scara --l1 0.7 --l2 0.6 --d 0.18 --q 0.0 --q 1.2 --q 0.05 -o forward_kinematics/out/scara_T.json
```

```bash
runroot python -m forward_kinematics.cli preset-wrist --type 1 --d7 0.12 --q 0.1 --q 0.2 --q 0.3 -o forward_kinematics/out/wrist_T.json
```

```bash
runroot python -m forward_kinematics.cli preset-wrist --type 2 --q 0.2 --q -0.4 --q 1.0 -o forward_kinematics/out/wrist2_T.json
```

```bash
runroot python -m forward_kinematics.cli preset-wrist --type 3 --q 0.0 --q 0.0 --q 0.0 -o forward_kinematics/out/wrist3_T.json
```

---

## 5) Class diagrams

```bash
runroot python -m forward_kinematics.cli diagram-dot -o forward_kinematics/out/classes.dot
```

```bash
runroot python -m forward_kinematics.tools.diagram plantuml --out classes.puml
```

```bash
runroot python -m forward_kinematics.tools.diagram graphviz --fmt png --dpi 300 --outstem classes
```

```bash
runroot python -m forward_kinematics.tools.diagram dot --theme dark --legend --out classes_dark.dot
```

```bash
runroot python -m forward_kinematics.tools.diagram plantuml --theme dark --legend --out forward_kinematics/out/classes_dark.puml
```

```bash
runroot python -m forward_kinematics.tools.diagram dot --no-cluster --out classes_nocluster.dot
```

---

## 6) Sphinx docs (no make / no Xcode required)

```bash
runroot python -m forward_kinematics.cli sphinx-skel docs
```

```bash
runroot python -m sphinx -b html docs docs/_build/html
```

```bash
runroot python -m sphinx.ext.apidoc -o docs/api forward_kinematics
```

```bash
runroot python -m sphinx -b html docs docs/_build/html
```

```bash
runroot python -m sphinx_autobuild docs docs/_build/html
```

---

## 7) Tests

```bash
runroot python -m pytest forward_kinematics/tests -q
```

```bash
runroot python -m pytest --maxfail=1 --disable-warnings -q
```

---

## 8) End-to-end (pick/modify as needed)

```bash
runroot python -m forward_kinematics.cli validate forward_kinematics/in/my_robot.json
```

```bash
runroot python -m forward_kinematics.cli fk forward_kinematics/in/my_robot.json --q 0.0 --q 0.0 --q 0.0 -o forward_kinematics/out/T.json
```

```bash
runroot python -m forward_kinematics.cli jacobian-space forward_kinematics/in/my_robot.json --q 0.0 --q 0.0 --q 0.0 -o forward_kinematics/out/J_space.json
```

```bash
runroot python -m forward_kinematics.cli jacobian-body forward_kinematics/in/my_robot.json --q 0.0 --q 0.0 --q 0.0 -o forward_kinematics/out/J_body.json
```

```bash
runroot python -m forward_kinematics.tools.diagram plantuml --out forward_kinematics/out/classes.puml
```

```bash
runroot python -m sphinx -b html docs docs/_build/html
```
