# Control Module — RUNS.md (Full CLI Command Catalog)

> **Scope:** Control techniques 14.1–14.118. All commands write outputs under `control/out/` and read inputs (if any) from `control/in/`.

## How to run

Prefer running from the **project root** using Python’s module mode:

```
python -m control.cli --help
python -m control.cli msd_pd --help
python -m control.cli pendulum_pid --help
python -m control.cli robot_ct --help
python -m control.cli diagram --help
```

If you *do* `cd control/`, you can still run module-mode by prefixing with `-m` and the package name from the project root. Script-mode (e.g., `python cli.py`) is not guaranteed in all IDEs; use `-m` to avoid import issues.

## Conventions

- Vectors are comma-separated without spaces, e.g. `--q "0.3,-0.2"`.
- Time horizon uses `--t` or `--T` (both accepted) in `msd_pd`, `--t/--T` in `pendulum_pid`.
- Files are written to `control/out/` automatically.
- Use `pytest control/tests -q` to validate the installation.

---

## 0) Quick sanity / help

```
python -m control.cli --help
python -m control.cli msd_pd --help
python -m control.cli pendulum_pid --help
python -m control.cli robot_ct --help
python -m control.cli diagram --help
```

---

## 1) Mass–Spring–Damper with PD (14.10 / 14.12–14.25)

### 1.1 Minimal run (defaults)
```
python -m control.cli msd_pd
```

### 1.2 Short horizon for smoke
```
python -m control.cli msd_pd --T 0.1
```

### 1.3 Custom plant
```
python -m control.cli msd_pd --m 2.0 --c 0.4 --k 20.0 --T 2.5
```

### 1.4 Different initial condition
```
python -m control.cli msd_pd --x0 "0.1,0.0" --T 3.0
```

### 1.5 Custom gains
```
python -m control.cli msd_pd --kp 50 --kd 12 --T 3.0
```

### 1.6 Export with custom name
```
python -m control.cli msd_pd --T 1.2 --out msd_pd_t12
```

### 1.7 Combined example
```
python -m control.cli msd_pd --m 1.5 --c 0.7 --k 12.0 --kp 40 --kd 9 --x0 "0.2,-0.1" --T 2.0 --out msd_combo
```

### 1.8 Using lowercase `--t` (equivalent)
```
python -m control.cli msd_pd --t 0.25 --out msd_lower_t
```

---

## 2) Nonlinear Pendulum + PID around θd=π/2 (14.80–14.88)

### 2.1 Minimal run (defaults)
```
python -m control.cli pendulum_pid
```

### 2.2 Short horizon
```
python -m control.cli pendulum_pid --T 1.0
```

### 2.3 Custom PID
```
python -m control.cli pendulum_pid --kp 25 --ki 3 --kd 8 --T 2.0 --out pend_pid_custom
```

### 2.4 Lowercase `--t` also accepted
```
python -m control.cli pendulum_pid --t 0.5 --out pend_t05
```

---

## 3) Planar 2R — Computed Torque PD (14.33 / 14.41 / 14.93–14.95)

> Uses the built-in analytic 2R dynamics. Inputs are desired/current joint states.

### 3.1 Zero-error, zero-accel (returns gravity torque at `q`)
```
python -m control.cli robot_ct --q "0.1,-0.2" --qd "0,0" --qd-d "0,0" --qdd-d "0,0"
```

### 3.2 Track desired velocity
```
python -m control.cli robot_ct --q "0.0,0.0" --qd "0.0,0.0" --qd-d "0.2,-0.1" --qdd-d "0.0,0.0"
```

### 3.3 Add desired acceleration
```
python -m control.cli robot_ct --q "0.0,0.0" --qd "0.0,0.0" --qd-d "0.0,0.0" --qdd-d "0.5,-0.3"
```

### 3.4 Tuning via (ζ, ωn)
```
python -m control.cli robot_ct --q "0.3,-0.2" --qd "0.0,0.0" --qd-d "0.0,0.0" --qdd-d "0.0,0.0" --wn 6.0 --zeta 0.9
```

---

## 4) Class Diagram (Mermaid Markdown)

### 4.1 Emit diagram to `control/out/classes.md`
```
python -m control.cli diagram
```

### 4.2 Emit with custom name
```
python -m control.cli diagram --out classes_control
```

Open the generated Markdown and preview Mermaid in your editor or docs site.

---

## 5) Tests (TDD)

```
pytest control/tests -q
```

Use verbose if needed:
```
pytest control/tests -q -vv
```

Run a single test:
```
pytest control/tests/test_pendulum_linearize_pid.py::test_linearize_matches_signs_and_dims -q
```

---

## 6) Typical outputs

- `control/out/msd_pd.json`
- `control/out/pend_pid.json`
- `control/out/classes.md`

Each JSON contains arrays: `t` (time) and `x` (states).

---

## 7) Notes

- All math maps to the chapter sections: MSD/PD (14.6–14.25), computed-torque PD (14.33/14.41/14.93–14.95), pendulum PID & linearization (14.80–14.88).
- For torque-bounded optimal control (14.118), consider adding a `control/optimal/` submodule using CasADi or pydrake DirectCollocation.
