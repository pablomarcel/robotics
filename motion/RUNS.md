# Robotics Motion Toolkit — **Run Commands (Fixed & Minimal)**

This doc lists commands that match the current CLI exactly.  
Every Python command is prefixed with `runroot` and **each command is in its own bash block**.

---

## -1) One-time session bootstrap

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
runroot mkdir -p motion/out motion/in
# -----------------------------------------------------------------------------
```

---

## 0) Environment

```bash
runroot python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
runroot python -m pip install --upgrade pip
```

```bash
runroot pip install numpy pytest pytest-sugar pytest-cov
```

```bash
# Optional backends:
runroot pip install graphviz
```

```bash
# Optional backends:
runroot pip install pylint
```

---

## 1) Tests

```bash
runroot pytest motion/tests -q
```

```bash
runroot pytest motion/tests/test_rotation.py -q
```

```bash
runroot pytest motion/tests/test_se3_inverse.py -k test_apply_point_and_apply_points -q
```

```bash
runroot pytest motion/tests --cov=motion --cov-report=term-missing
```

---

## 2) CLI (available commands)

```bash
runroot python -m motion.cli --help
```

> Subcommands: `rotation | screw | plucker | lines | plane-dist | fk | run`

### rotation

```bash
runroot python -m motion.cli rotation --axis 0,0,1 --angle 1.57079632679
```

```bash
runroot python -m motion.cli rotation --axis 0,0,1 --angle 90 --degrees
```

### screw

```bash
runroot python -m motion.cli screw --axis 0,0,1 --s 0,0,0 --pitch 0.5 --phi 1.57079632679
```

```bash
runroot python -m motion.cli screw --axis 0,0,1 --s 0,0,0 --pitch 0.5 --phi 90 --degrees
```

### plucker

```bash
runroot python -m motion.cli plucker --p1 0,0,0 --p2 1,0,0
```

### lines

```bash
runroot python -m motion.cli lines --a1 0,0,0 --a2 1,0,0 --b1 0,1,1 --b2 0,2,3
```

### plane-dist

```bash
runroot python -m motion.cli plane-dist --point 0,0,1 --normal 0,0,1 --s 0
```

```bash
runroot python -m motion.cli plane-dist --point 0,0,1 --normal 0,0,1 --s 0 --unsigned
```

### fk

```bash
runroot python -m motion.cli fk --dh 0.1,0.0,0.2,0.0 --dh 0.3,1.5707963,0.0,0.5
```

### run (file-driven)

```bash
runroot python -m motion.cli run --file motion/in/job.json
```

---

## 3) IO service (example)

```bash
runroot python - <<'PY'
from motion.io import IO
import numpy as np
io = IO()
T = np.eye(4)
io.save_transform(T, "pose_A")
print("Wrote motion/out/pose_A.json")
PY
```

---

## 4) Diagrams (skip backends if unavailable)

```bash
runroot python -m motion.tools.diagram discover --package motion
```

```bash
runroot python -m motion.tools.diagram json --out classes.json
```

```bash
runroot python -m motion.tools.diagram plantuml --out classes.puml
```

```bash
runroot python -m motion.tools.diagram mermaid --out classes.mmd
```

```bash
runroot python -m motion.tools.diagram all
```

### Optional rendering (no system Graphviz required)

```bash
docker run --rm -v "$PWD":/work -w /work minlag/mermaid-cli mmdc -i motion/out/classes.mmd -o motion/out/classes.png
```

```bash
docker run --rm -v "$PWD":/work -w /work plantuml/plantuml -tpng motion/out/classes.puml
```

---

## 5) End-to-end (sample inputs assumed under motion/in)

```bash
runroot python -m motion.cli screw --axis 0,0,1 --s 0,0,0 --pitch 0.2 --phi 1.0 --out motion/out/screw_T.json
```

```bash
runroot python -m motion.cli lines --a1 0,0,0 --a2 1,0,0 --b1 0,1,1 --b2 0,2,3 --out motion/out/plucker_result.json
```

```bash
runroot python -m motion.cli fk --dh 0.1,0.0,0.2,0.0 --dh 0.3,1.5707963,0.0,0.5 --out motion/out/fk_T.json
```

```bash
runroot python -m motion.cli run --file motion/in/job.json --out motion/out/job_result.json
```

---

## 6) Cleanups

```bash
runroot rm -rf motion/out/*
```

```bash
runroot ls -1 motion/in
```

```bash
runroot ls -1 motion/out
```
