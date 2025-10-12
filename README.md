# Robotics (Python) — CLI‑First Study & Design Suite

> **Mission:** prove you can learn, reproduce, and *do* robot kinematics, dynamics, path planning, and control **without MATLAB®/Simulink®** — using Python, a modern CLI workflow, and open libraries.

This repo is a collection of focused, test‑driven Python packages (“tools”) that replicate and extend core workflows from **Reza N. Jazar, _Theory of Applied Robotics_ (3rd ed.)**. Each package ships with a friendly **CLI**, example inputs, and a **RUNS.md** full of copy‑paste commands. No notebooks required, no proprietary stack needed.

<p align="center">
  <em>“MATLAB® is not a skill.”</em>
</p>

---

## Why this exists

- I did not have a MATLAB® license - and I do not need one (and don't want one).
- Python’s ecosystem (NumPy, SciPy, SymPy, python‑control, etc.) does everything the textbooks require — but it’s code‑heavy.
- So I wrapped the hard parts into clean **command‑line tools** with consistent I/O, file conventions, and tests.
- The result is a **drop‑in study companion** and **reproducible design lab** for robotics.

---

## What’s inside — ordered like Jazar’s book

Each subfolder is a cohesive package with its own CLI, tests, and a RUNS.md. Short blurbs are shown where the package is already implemented; placeholders are noted where work is pending.

> Folder names are short and pragmatic (what you type at the CLI), but the order below mirrors the book’s table of contents.

### 1) Introduction (`intro/`)  — _WIP_
- CLI scaffolds and helper tools for basic signals/blocks used throughout (e.g., triads, vectors, frames).

---

### Part I — Kinematics

#### 2) Rotation Kinematics (`rotation/`)
- Global/local axis rotations, successive rotations, Euler angles, local vs global, general transformations, active vs passive maps.

#### 3) Orientation Kinematics (`orientation/`)
- Axis–angle, rotation matrices, Euler parameters/quaternions, spinors/rotators, representation caveats, composition/decomposition of rotations.

#### 4) Motion Kinematics (`motion/`)
- Rigid body motion, homogeneous transforms, order‑free & screw transformations, Plücker coordinates, lines/planes, combined screw operations.

#### 5) Forward Kinematics (`forward/`)
- Denavit–Hartenberg, adjacent frame transforms, forward pose of manipulators, spherical wrist, assembly strategies, screw‑based transforms, non‑DH methods.

#### 6) Inverse Kinematics (`inverse/`)
- Decoupling, inverse transforms, iterative methods, existence/uniqueness, singularities, technique comparison.

---

### Part II — Derivative Kinematics

#### 7) Angular Velocity (`angular/`)
- Angular velocity vectors/matrices, time‑derivatives across frames, rigid‑body ω, velocity transform matrices, ∂/∂t of T.

#### 8) Velocity Kinematics (`velocity/`)
- Rigid link and forward velocity, Jacobian generating vectors, inverse velocity kinematics, linear/nonlinear algebraic formulations, Jacobian from link transforms.

#### 9) Acceleration Kinematics (`acceleration/`)
- Angular & rigid‑body acceleration, acceleration transform matrices, forward/inverse acceleration, recursive acceleration, higher‑order frame derivatives.

---

### Part III — Dynamics

#### 10) Applied Dynamics (`applied/`)
- Forces/moments, momentum, equations of motion, work/energy, translational/rotational kinetics, mass moment matrices, Lagrange/Newton forms.

#### 11) Robot Dynamics (`robot/`)
- Rigid‑link Newton–Euler and recursive Newton–Euler, Lagrange dynamics, statics, Lagrange equations with link transform matrices.

---

### Part IV — Control

#### 12) Path Planning (`path/`)
- Cubic/polynomial/non‑polynomial paths, spatial design, forward/inverse path robot motion, rotational paths.

#### 13) Time‑Optimal Control (`timeopt/`)
- Minimum‑time & bang‑bang, floating time, time‑optimal control for robots.

#### 14) Control Techniques (`control/`)
- Open/closed‑loop control, computed torque, linear control (P, I, D), sensing & control (position/speed/acceleration sensors).

---

## Design philosophy

- **CLI‑first**: everything important is a flag, not hidden in a notebook cell.
- **Reproducible**: inputs in `in/`, outputs in `out/`, commands in `RUNS.md`.
- **Test‑driven**: `pytest` suites for each tool; coverage reports during refactors.
- **Pragmatic math**: uses python‑control and explicit numerics/symbolics where appropriate for clarity and robustness.
- **Teacher‑friendly**: plots (Matplotlib and Plotly), CSV/JSON exports, and clean logs that drop into lectures/reports.

---

## Quick start

```bash
# 1) Clone and create a virtual env (example)
git clone https://github.com/pablomarcel/robotics.git
cd robotics
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -U pip
pip install -r requirements.txt

# 3) Run a demo from *inside* a package
cd rotation
python cli.py --help
# See RUNS.md in each package for copy‑paste commands

# 4) Run tests (per tool)
cd ../
pytest rotation/tests --cov --cov-config=rotation/.coveragerc --cov-report=term-missing
```

> Each folder includes an import shim so that **`python cli.py ...` works when you `cd` into that folder**.

---

## I/O conventions

- **Inputs**: `in/` (JSON/CSV/YAML)
- **Outputs**: `out/` (CSV/JSON/HTML/PNG)
- Many CLIs support `--pretty`, `--save_json`, `--save_csv`, and `--save` with placeholders like `out/{{name}}_{{kind}}.png` or `.html`.

Common capabilities:
- print numeric results to console
- export matrices, Jacobians, and trajectories to **CSV/JSON**
- write **Plotly** interactive HTML and **Matplotlib** PNGs

---

## Example: rotation composition (rotation/)

```bash
cd rotation
python cli.py compose --about global --angles "30, -20, 45" --order "z,y,x" --plot --save "out/compose_{{kind}}.png"
```

---

## Tested setup

- Python 3.13 (also works with 3.11/3.12 in most tools)
- NumPy 2.x, SciPy 1.15.x, SymPy 1.13.x, matplotlib 3.10.x, plotly 5.x
- macOS 13.7, Windows 10/11 (CLI + plots)
- Continuous refactors with `pytest` suites per tool

See `requirements.txt` for exact pins.

---

## Trademark & affiliation notice

**MATLAB®** and **Simulink®** are registered trademarks of **The MathWorks, Inc.** I am **not affiliated** with The MathWorks, Reza N. Jazar, or the publisher. Book references are for citation and interoperability only.

---

## Contributing

Issues and PRs are welcome:
- Respect the folder structure and the **CLI‑first** approach.
- Keep inputs in `in/`, outputs in `out/`, and runnable **RUNS.md** examples.
- Add or update **tests** with any new feature or refactor.
- Prefer small, focused modules and clean dataclasses for I/O (`apis.py`).

A simple PR checklist:
- [ ] `pytest` passes locally for the changed tool(s)
- [ ] `RUNS.md` updated with new/changed commands
- [ ] New flags documented in `cli.py --help`
- [ ] Outputs reproducible under `out/`

---

## License

This project is released under the **MIT License** (see `LICENSE`).

---

## Acknowledgments

- Reza N. Jazar, _Theory of Applied Robotics_ (3rd ed.)
- The Python open‑source ecosystem: NumPy, SciPy, SymPy, matplotlib, plotly, python‑control, and many others.
