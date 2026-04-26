# Control Module — RUNS.md (Clean Catalog)

> Commands use `runroot` and write under `control/out/`. Inputs (if any) are read from `control/in/`.

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
runroot mkdir -p control_techniques/out
# -----------------------------------------------------------------------------
```

---

## 0) Quick sanity / help

```bash
runroot python -m control_techniques.cli --help
```

```bash
runroot python -m control_techniques.cli msd_pd --help
```

```bash
runroot python -m control_techniques.cli pendulum_pid --help
```

```bash
runroot python -m control_techniques.cli robot_ct --help
```

```bash
runroot python -m control_techniques.cli diagram --help
```

---

## 1) Mass–Spring–Damper with PD

```bash
runroot python -m control_techniques.cli msd_pd
```

```bash
runroot python -m control_techniques.cli msd_pd --T 0.1
```

```bash
runroot python -m control_techniques.cli msd_pd --m 2.0 --c 0.4 --k 20.0 --T 2.5
```

```bash
runroot python -m control_techniques.cli msd_pd --x0 "0.1,0.0" --T 3.0
```

```bash
runroot python -m control_techniques.cli msd_pd --kp 50 --kd 12 --T 3.0
```

```bash
runroot python -m control_techniques.cli msd_pd --T 1.2 --out msd_pd_t12
```

```bash
runroot python -m control_techniques.cli msd_pd --m 1.5 --c 0.7 --k 12.0 --kp 40 --kd 9 --x0 "0.2,-0.1" --T 2.0 --out msd_combo
```

```bash
runroot python -m control_techniques.cli msd_pd --t 0.25 --out msd_lower_t
```

---

## 2) Pendulum PID (θd = π/2)

```bash
runroot python -m control_techniques.cli pendulum_pid
```

```bash
runroot python -m control_techniques.cli pendulum_pid --t 1.0
```

```bash
runroot python -m control_techniques.cli pendulum_pid --kp 25 --ki 3 --kd 8 --t 2.0 --out pend_pid_custom
```

```bash
runroot python -m control_techniques.cli pendulum_pid --t 0.5 --out pend_t05
```

---

## 3) Planar 2R — Computed Torque PD

```bash
runroot python -m control_techniques.cli robot_ct --q "0.1,-0.2" --qd "0,0" --qd-d "0,0" --qdd-d "0,0"
```

```bash
runroot python -m control_techniques.cli robot_ct --q "0.0,0.0" --qd "0.0,0.0" --qd-d "0.2,-0.1" --qdd-d "0.0,0.0"
```

```bash
runroot python -m control_techniques.cli robot_ct --q "0.0,0.0" --qd "0.0,0.0" --qd-d "0.0,0.0" --qdd-d "0.5,-0.3"
```

```bash
runroot python -m control_techniques.cli robot_ct --q "0.3,-0.2" --qd "0.0,0.0" --qd-d "0.0,0.0" --qdd-d "0.0,0.0" --wn 6.0 --zeta 0.9
```

---

## 4) Class Diagram

```bash
runroot python -m control_techniques.cli diagram
```

```bash
runroot python -m control_techniques.cli diagram --out classes_control
```

### Sphinx

python -m control_techniques.cli sphinx-skel control_techniques/docs
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html
sphinx-autobuild docs docs/_build/html