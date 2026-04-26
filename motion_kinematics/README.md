# Robotics Motion Toolkit — Run Commands Cheat Sheet

This README collects **all practical run commands** for this project, grouped by task.
It favors copy‑pasteable snippets and notes optional tools (Graphviz/pyreverse)
that may be unavailable locally.

> Project layout (key files):
>
> ```
> motion/
>   app.py
>   apis.py
>   cli.py
>   core.py
>   design.py
>   io.py
>   utils.py
>   tools/
>     diagram.py
>   in/        # inputs
>   out/       # outputs
>   tests/     # pytest suite
> ```

---

## 0) Setup

### Create & activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### Install Python dependencies
```bash
pip install numpy pytest pytest-sugar pytest-cov
# Optional backends (only if you want them):
pip install graphviz           # Python bindings (still needs 'dot' executable to render)
pip install pylint             # Provides 'pyreverse'
```

### (Optional) Make the package importable during development
```bash
export PYTHONPATH="$PWD"
# Windows PowerShell: $env:PYTHONPATH="$PWD"
```

---

## 1) Running the Test Suite (TDD)

### Run all tests
```bash
pytest motion_kinematics/tests -q
```

### Run a single test file
```bash
pytest motion_kinematics/tests/test_rotation.py -q
```

### Run a specific test function
```bash
pytest motion_kinematics/tests/test_se3_inverse.py -k test_apply_point_and_apply_points -q
```

### Show coverage
```bash
pytest motion_kinematics/tests --cov=motion_kinematics --cov-report=term-missing
```

### Typical quick filters
```bash
# Only utils:
pytest motion_kinematics/tests/test_utils.py -q
# Only geometry core:
pytest motion_kinematics/tests/test_*se3*.py -q
```

---

## 2) Command Line Interface (CLI)

All CLI entry points assume the current directory is the project root.

### Show CLI help
```bash
python -m motion_kinematics.cli --help
```

> Each subcommand also provides help, e.g.:
```bash
python -m motion_kinematics.cli se3 --help
python -m motion_kinematics.cli rotation_kinematics --help
python -m motion_kinematics.cli screw --help
python -m motion_kinematics.cli plucker --help
python -m motion_kinematics.cli plane --help
python -m motion_kinematics.cli dh --help
```

### Rotation examples
```bash
# Build R from axis-angle (axis=0,0,1; phi=1.5708 rad) and print the 3x3:
python -m motion_kinematics.cli rotation_kinematics axis-angle --axis 0 0 1 --phi 1.57079632679

# Canonical rotations:
python -m motion_kinematics.cli rotation_kinematics rz --theta 1.2
python -m motion_kinematics.cli rotation_kinematics rx --theta 0.5
python -m motion_kinematics.cli rotation_kinematics ry --theta -0.7
```

### SE(3) examples
```bash
# Build SE3 from Rz(90deg) and t=[1,2,3], print 4x4 and apply to a point
python -m motion_kinematics.cli se3 from-rz --theta 1.57079632679 --t 1 2 3
python -m motion_kinematics.cli se3 apply --T "[[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]]" --p 1 0 0

# Inverse and compose
python -m motion_kinematics.cli se3 inverse --T "[[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]]"
python -m motion_kinematics.cli se3 compose --A "<4x4 json>" --B "<4x4 json>"
```

### Screw motion examples
```bash
# Screw with axis u=(0,0,1), point s=(0,0,0), pitch h=0.5, angle phi=pi/2
python -m motion_kinematics.cli screw to-matrix --u 0 0 1 --s 0 0 0 --h 0.5 --phi 1.57079632679
```

### Plücker line examples
```bash
# Build two lines from points, compute angle & distance
python -m motion_kinematics.cli plucker angle-distance --p1 0 0 0 --p2 1 0 0 --q1 0 1 1 --q2 0 2 3
# Transform a line by SE(3)
python -m motion_kinematics.cli plucker transform --p1 0 0 0 --p2 1 0 0 --T "[[1,0,0,0.5],[0,1,0,-0.2],[0,0,1,0.3],[0,0,0,1]]"
```

### Plane examples
```bash
# Plane through point (0,0,1) with normal (0,0,1), distance of p=(0,0,2)
python -m motion_kinematics.cli plane distance --point 0 0 1 --normal 0 0 1 --p 0 0 2
```

### DH forward kinematics
```bash
# Provide a small DH table (rows of a alpha d theta); prints final SE(3)
python -m motion_kinematics.cli dh fk --csv motion_kinematics/in/my_dh.csv
# Or inline JSON:
python -m motion_kinematics.cli dh fk --json "[[0.1, 0.0, 0.2, 0.0],[0.3, 1.5707963, 0.0, 0.5]]"
```

> All CLI commands accept `--out` flags when applicable to save artifacts under `motion/out`.

---

## 3) I/O Helpers

The `IO` service centralizes atomic read/write under `motion/in` and `motion/out`.

### Quick usage (Python REPL or script)

```python
from motion_kinematics.io import IO
import numpy as np

io = IO()
T = np.eye(4)
io.save_transform(T, "pose_A")  # writes motion_kinematics/out/pose_A.npy and .json
P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
io.save_points_csv(P, "cloud.csv")  # writes motion_kinematics/out/cloud.csv
loaded = io.load_points_csv("cloud.csv")  # reads from motion_kinematics/in by default
```

---

## 4) Diagram Generation (no system Graphviz required)

### Discover & JSON model
```bash
python -m motion_kinematics.tools.diagram discover --package motion_kinematics
python -m motion_kinematics.tools.diagram json --out motion_kinematics/out/classes.json
```

### PlantUML & Mermaid sources
```bash
python -m motion_kinematics.tools.diagram plantuml --out motion_kinematics/out/classes.puml
python -m motion_kinematics.tools.diagram mermaid  --out motion_kinematics/out/classes.mmd
```

### Render-all convenience (skips where unavailable)
```bash
python -m motion_kinematics.tools.diagram all
# Produces: JSON, PlantUML, Mermaid; attempts Graphviz/pyreverse if available.
```

> **Note:** If `dot` (Graphviz CLI) is not installed, Graphviz PNG/SVG rendering will be skipped.
> If `pyreverse` is not available or fails, it will be skipped as well.

#### Optional: Render Mermaid to PNG (no Graphviz)
- Using Dockerized Mermaid CLI:
```bash
docker run --rm -v "$PWD":/work -w /work minlag/mermaid-cli   mmdc -i motion_kinematics/out/classes.mmd -o motion_kinematics/out/classes.png
```

- Using Node (if available):
```bash
npm -g install @mermaid-js/mermaid-cli
mmdc -i motion_kinematics/out/classes.mmd -o motion_kinematics/out/classes.png
```

#### Optional: Render PlantUML
- Via a PlantUML server (upload `motion/out/classes.puml`).
- Or Docker:
```bash
docker run --rm -v "$PWD":/work -w /work plantuml/plantuml   -tpng motion_kinematics/out/classes.puml
```

---

## 5) Examples: End-to-End CLI runs

### Compute angle & distance between skew lines and save to JSON
```bash
python -m motion_kinematics.cli plucker angle-distance   --p1 0 0 0 --p2 1 0 0   --q1 0 1 1 --q2 0 2 3   --out motion_kinematics/out/plucker_result.json
```

### Build a screw transform and apply it to points loaded from CSV
```bash
# points in: motion_kinematics/in/points.csv  (Nx3, no header)
python -m motion_kinematics.cli screw to-matrix --u 0 0 1 --s 0 0 0 --h 0.2 --phi 1.0 --out motion_kinematics/out/screw_T.json
python -m motion_kinematics.cli se3 apply-points   --T motion_kinematics/out/screw_T.json   --points motion_kinematics/in/points.csv   --out motion_kinematics/out/points_transformed.csv
```

### DH forward kinematics from CSV
```bash
python -m motion_kinematics.cli dh fk --csv motion_kinematics/in/dh_table.csv --out motion_kinematics/out/fk_T.json
```

---

## 6) Troubleshooting & Tips

- **Import errors**: ensure `export PYTHONPATH="$PWD"` (or install as a package).
- **Graphviz errors**: Python `graphviz` package alone is not enough; `dot` must be on PATH to render. If not available, prefer Mermaid/PlantUML commands above.
- **pyreverse issues**: Provided by `pylint`. If present but failing, use the `diagram all` command; it will continue with other outputs.

---

## 7) Useful One-Liners

```bash
# List input/output files
ls -1 motion_kinematics/in
ls -1 motion_kinematics/out

# Clean outputs
rm -rf motion_kinematics/out/*

# Run only geometry tests, verbose
pytest motion_kinematics/tests -k "plucker or se3 or rotation" -vv
```

---

## 8) Version

This toolkit follows semantic-ish versioning reported by:
```bash
python - <<'PY'
from motion.utils import version_string
print(version_string())
PY
```
