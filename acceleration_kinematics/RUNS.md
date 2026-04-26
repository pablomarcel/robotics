# RUNS.md — App Run Commands (clean & focused)

This document lists **only** the commands you need to **run the app** from repository root.
We intentionally exclude all test/pytest commands (those live elsewhere).

Target interpreter: **Python 3.11**.

> Tip: The helper `runroot` lets you run a command from the project root without changing your shell’s cwd.

---

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
runroot mkdir -p acceleration_kinematics/out
# -----------------------------------------------------------------------------
```

---

## 0) Verify environment & install the package

**Show Python version**
```bash
runroot python --version
```

**Upgrade base tooling**
```bash
runroot python -m pip install --upgrade pip setuptools wheel
```

**Editable install of the package**
```bash
runroot python -m pip install -e .
```

**(Optional) Install from requirements file (if present)**
```bash
runroot python -m pip install -r requirements.txt
```

**Ensure the output directory exists (Python one-liner)**
```bash
runroot python -c "import pathlib; pathlib.Path('acceleration/out').mkdir(parents=True, exist_ok=True)"
```

> Note: If you use a virtual environment, activate it in your shell as usual
> (e.g., `source .venv/bin/activate` on Unix/macOS) before running the commands above.

---

## 1) CLI usage via module runner (works regardless of console-script name)

**Show CLI help**
```bash
runroot python -m acceleration_kinematics.cli --help
```

**Show help for a subcommand (example: `diagram`)**
```bash
runroot python -m acceleration_kinematics.cli diagram --help
```

**Generate a Mermaid diagram (example path)**
```bash
runroot python -m acceleration_kinematics.cli diagram --output acceleration_kinematics/out/diagram.mmd
```

---

## 2) Discover installed console scripts (if any)

**List console scripts exposed by the installed package**
```bash
runroot python -c "import importlib.metadata as m; print('\n'.join(ep.name for ep in m.entry_points().select(group='console_scripts')))"
```

---

## 3) Quick library sanity checks

**Compute a rotation matrix from ZYX Euler angles**
```bash
runroot python - <<'PY'
from acceleration.tools.euler import euler_matrix
R = euler_matrix('ZYX', [0.1, 0.2, 0.3])
print(R)
PY
```

**Map ZYX Euler rates to angular velocity**
```bash
runroot python - <<'PY'
import numpy as np
from acceleration.tools.euler import euler_rates_matrix
E = euler_rates_matrix('ZYX', [0.1, 0.2, 0.3])
qd = np.array([0.01, 0.02, -0.03])
print('omega =', E @ qd)
PY
```

**Compute classic rigid-body acceleration for a point offset**
```bash
runroot python - <<'PY'
import numpy as np
from acceleration.utils import classic_accel
alpha = np.array([0.1, -0.2, 0.3])
omega = np.array([1.0, 0.5, -0.4])
r     = np.array([0.2, -0.1, 0.05])
print('a =', classic_accel(alpha, omega, r))
PY
```

---

## 4) Packaging (optional)

**Install build backend**
```bash
runroot python -m pip install build
```

**Build wheel and sdist**
```bash
runroot python -m build
```

**Print absolute path to the `dist/` folder**
```bash
runroot python - <<'PY'
from pathlib import Path; print(Path('dist').resolve())
PY
```

---

## 5) Troubleshooting quick checks

**Show where the package is imported from**
```bash
runroot python - <<'PY'
import acceleration, inspect, os
print(os.path.dirname(inspect.getfile(acceleration)))
PY
```

**Print environment details (Python, NumPy, platform)**
```bash
runroot python - <<'PY'
import platform, sys
print('Python:', sys.version)
print('Platform:', platform.platform())
try:
    import numpy as np
    print('NumPy:', np.__version__)
except Exception as e:
    print('NumPy: not installed', e)
PY
```

### Sphinx

python -m acceleration_kinematics.cli sphinx-skel acceleration_kinematics/docs
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html
sphinx-autobuild docs docs/_build/html