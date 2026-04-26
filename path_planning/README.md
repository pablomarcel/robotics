# Robotics Path Planning — Run Commands

This README lists **all the practical run commands** you’ll need for this project:
- how to set up a virtual environment
- how to run the unit tests (TDD)
- how to use the programmatic API from Python
- how to run the optional HTTP API server
- how to call the HTTP endpoints with `curl`
- how to generate a class diagram from the codebase
- how to export sample trajectories to files

> Project layout (key parts):
>
> ```text
> path/
>   app.py            # High-level facade used by tests and scripts
>   apis.py           # Programmatic + optional FastAPI HTTP API
>   core.py           # Base classes (BoundaryConditions, Trajectory interfaces, etc.)
>   poly.py           # Cubic/Quintic/Septic/LeastSquares polynomials
>   spatial.py        # ParabolicBlend3D, Harmonic1D, Cycloid1D, ComposeYofX
>   time.py           # Time laws (LSPB, QuinticTime)
>   robot.py          # Simple Planar 2R manipulator model
>   rotation.py       # Angle-axis rotation path
>   io.py             # IOManager (in/out helpers)
>   tests/            # pytest suite (TDD)
>   in/               # input files
>   out/              # output files
> ```
>
> All IO is explicit and normally goes under `path/in` and `path/out`.

---

## 0) Environment Setup

> Requires Python 3.11+

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# Minimal dev set (tests + optional HTTP)
pip install pytest pytest-sugar pytest-xdist coverage
pip install fastapi uvicorn httpx     # optional HTTP server + test client

# Numerical stack (if not present)
pip install numpy scipy sympy

# Diagram tooling (optional but recommended)
pip install py2puml plantuml graphviz
```

If your system lacks a local `plantuml` CLI, you can still produce the `.puml` file with `py2puml` and render it later (or use any PlantUML renderer plugin).

---

## 1) Run the Tests (TDD)

```bash
# Run the whole suite
pytest path_planning/tests -q

# Verbose
pytest path_planning/tests -vv

# Focus a subset
pytest path_planning/tests -q -k poly

# Parallel (xdist)
pytest path_planning/tests -q -n auto

# Coverage (HTML report in htmlcov/)
coverage run -m pytest path_planning/tests -q
coverage html
```

---

## 2) Programmatic API (Python)

### 2.1 Polynomials

```bash
python - <<'PY'
import numpy as np
from path.apis import poly_api

# Quintic example
resp = poly_api(kind="quintic", t0=0, tf=1, q0=10, qf=45, samples=5)
print(resp.keys())
print("q(t) endpoints:", resp["q"][0], "->", resp["q"][-1])
PY
```

### 2.2 2R Inverse Kinematics Along a Line

```bash
python - <<'PY'
import numpy as np
from path.apis import ik2r_api
resp = ik2r_api(l1=0.25, l2=0.25, path_type="line", x0=0.2, y0=0.1, x1=0.1, y1=0.2, samples=20)
print("th1 len:", len(resp["th1"]), "th2 len:", len(resp["th2"]))
PY
```

### 2.3 Rotation Path (Angle-Axis)

```bash
python - <<'PY'
import numpy as np
from path.apis import rot_api
R0 = np.eye(3).tolist()
Rf = np.diag([-1,-1,1]).tolist()  # 180deg about Z
resp = rot_api(R0, Rf, samples=5)
print("Got", len(resp["R"]), "rotation matrices")
PY
```

---

## 3) Optional HTTP Server

The HTTP layer delegates validation to the programmatic API for maximum compatibility.

### 3.1 Start the server (dev reload)

```bash
uvicorn path_planning.apis:get_http_app --reload --port 8000
```

### 3.2 Curl the endpoints

#### `/poly`

```bash
curl -s http://localhost:8000/poly   -H 'Content-Type: application/json'   -d '{"kind":"cubic","t0":0,"tf":1,"q0":10,"qf":45,"samples":5}' | jq .
```

#### `/ik2r`

```bash
curl -s http://localhost:8000/ik2r   -H 'Content-Type: application/json'   -d '{"l1":0.25,"l2":0.25,"path_type":"line","x0":0.2,"y0":0.1,"x1":0.1,"y1":0.2,"samples":20}' | jq .
```

#### `/rot`

```bash
curl -s http://localhost:8000/rot   -H 'Content-Type: application/json'   -d '{"R0":[[1,0,0],[0,1,0],[0,0,1]],"Rf":[[-1,0,0],[0,-1,0],[0,0,1]],"samples":5}' | jq .
```

> If you don’t have `jq`, you can omit `| jq .`

---

## 4) Exporting Trajectories to Files

Use the `PathPlannerApp` with `IOManager` to write to `path/out`:

```bash
python - <<'PY'
import numpy as np
from path.app import PathPlannerApp
from path.core import BoundaryConditions

app = PathPlannerApp()

# Quintic example
bc = BoundaryConditions(0, 1, 10, 45, 0, 0, 0, 0)
traj = app.quintic(bc)
t = np.linspace(0, 1, 25)
samp = app.sample_1d(traj, t)

# Save CSV into path/out/
app.io.ensure_dirs()
out_file = app.io.save_csv("quintic_sample.csv", {"t":samp.t, "q":samp.q, "qd":samp.qd, "qdd":samp.qdd})
print("Wrote:", out_file)
PY
```

---

## 5) Generate Class Diagram (PlantUML)

Two options are shown; pick whichever you have tools for.

### 5.1 Using `py2puml`

```bash
# Generate PlantUML from Python packages
py2puml path_planning path_planning/out/diagram.puml

# Render PNG/SVG (requires local plantuml CLI)
plantuml -tpng path_planning/out/diagram.puml    # PNG
plantuml -tsvg path_planning/out/diagram.puml    # SVG
```

### 5.2 Using `plantuml` only (if you already have a .puml)

```bash
plantuml -tpng path_planning/out/diagram.puml
```

> Tip: many editors preview `.puml` directly via extensions.

---

## 6) Reproducible Examples for Textbook Eqs (12.1–12.301)

Most textbook paths are covered by the following classes/methods:
- **Cubic/Quintic/Septic** (boundary-value polynomials)
- **LSPB** (trapezoidal/triangular velocity profiles)
- **Cycloid** and **Harmonic** (rest-to-rest motion laws)
- **ParabolicBlend3D** (corner blend for piecewise-linear spatial paths)
- **Planar2R** (FK/IK) and **AngleAxisPath** (SO(3) interpolation)

You can assert the exact polynomial coefficients and sampled values via:
```bash
python - <<'PY'
import numpy as np
from path.app import PathPlannerApp
from path.core import BoundaryConditions

app = PathPlannerApp()
res = app.poly_path(q0=10, qf=45, t0=0, tf=1, order=5, qd0=0, qdf=0, qdd0=0, qddf=0, samples=11)
print("coeffs:", res["a"])
print("q[0], q[-1]:", res["q"][0], res["q"][-1])
PY
```

---

## 7) Troubleshooting

- If the FastAPI smoke test fails with a 422, ensure you used the **raw dict** accepting endpoints provided here (they delegate validation to the programmatic API).
- Numpy version must be recent (≥ 2.0 recommended). If you see numeric drift in jerk tests, confirm `numpy` and `pytest` are up-to-date.
- On macOS, you may need: `brew install graphviz plantuml` for diagram rendering.

---

## 8) Handy One-Liners

```bash
# Run every test and stop on first failure
pytest path_planning/tests -x -q

# Only run FastAPI smoke test (if httpx/fastapi installed)
pytest path_planning/tests -q -k http_fastapi_smoke

# Serve API on port 9000 (no reload)
uvicorn path_planning.apis:get_http_app --host 0.0.0.0 --port 9000

# Generate diagram .puml and PNG in path_planning/out
mkdir -p path_planning/out && py2puml path_planning path_planning/out/diagram.puml && plantuml -tpng path_planning/out/diagram.puml

# Export sample LSPB trajectory
python - <<'PY'
import numpy as np
from path.app import PathPlannerApp
from path.core import BoundaryConditions
app = PathPlannerApp()
bc = BoundaryConditions(0,1,0,1)
traj = app.lspb(bc, vmax=1.2)   # or amax=...
t = np.linspace(0,1,101)
s = app.sample_1d(traj, t)
app.io.ensure_dirs()
print(app.io.save_csv("lspb.csv", {"t":s.t, "q":s.q, "qd":s.qd, "qdd":s.qdd}))
PY
```

---

**Enjoy building paths ✨**