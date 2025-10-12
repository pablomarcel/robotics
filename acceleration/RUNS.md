# Robotics Project — Run Commands (Full & Exhaustive)

This README lists **ready-to-copy** commands for the most common developer tasks: environment setup, test runs, coverage, formatting, linting, type checking, packaging, and CLI usage. Commands are grouped and include variants (quiet/verbose, single-test, parallel, etc.).

> Target runroot python: **3.11** (per test logs). Replace `runroot python` with `runroot python3` if your system needs it.

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
runroot mkdir -p acceleration/out
# -----------------------------------------------------------------------------
```

---

## 1) Create & activate a virtual environment

### macOS / Linux (bash / zsh)

```bash
runroot python -m venv .venv
source .venv/bin/activate
runroot python -m pip install --upgrade pip setuptools wheel
```

### Windows (PowerShell)

```powershell
runroot python -m venv .venv
.\.venv\Scripts\Activate.ps1
runroot python -m pip install --upgrade pip setuptools wheel
```

---

## 2) Install dependencies

> Use one of the following patterns depending on what the repo provides.

### a) Editable install with dev extras (preferred if available)

```bash
pip install -e ".[dev]"
```

### b) Requirements files

```bash
pip install -r requirements.txt
# Optional dev/test tools (if provided)
pip install -r requirements-dev.txt
```

### c) Minimal install (if no extras/requirements are present)

```bash
pip install pytest pytest-sugar pytest-xdist anyio pytest-cov
```

---

## 3) Run the test suite

### Full test run (matching your logs)

```bash
pytest acceleration/tests
```

### Quiet output (just dots / minimal)

```bash
pytest acceleration/tests -q
```

### Verbose output

```bash
pytest acceleration/tests -vv
```

### Filter by keyword (only tests with "euler_quat")

```bash
pytest acceleration/tests -k euler_quat -vv
```

### Run a single test file

```bash
pytest acceleration/tests/test_classic_accel.py -vv
```

### Run a single test (node id)

```bash
pytest acceleration/tests/test_classic_accel.py::test_accel_from_euler_derivatives_in_inertial_frame -vv
```

### Stop after first failure

```bash
pytest acceleration/tests -x
```

### Re-run last failures only

```bash
pytest acceleration/tests --last-failed -vv
```

### Show durations (slowest tests)

```bash
pytest acceleration/tests --durations=10 -vv
```

### Run in parallel (auto detect cores)

```bash
pytest acceleration/tests -n auto
```

---

## 4) Coverage

### Quick coverage over the whole suite

```bash
pytest acceleration/tests --cov=acceleration --cov-report=term-missing -vv
```

### Generate HTML coverage report

```bash
pytest acceleration/tests --cov=acceleration --cov-report=html -vv
# Then open:
runroot python -c "import webbrowser, pathlib; webbrowser.open_new_tab(pathlib.Path('htmlcov/index.html').resolve().as_uri())"
```

---

## 5) Code formatting & linting

> Use whichever tools your repo includes. The commands below are standard.

### Black (format)

```bash
black .
```

### Black (check only, no changes)

```bash
black --check .
```

### Ruff (lint + fixes)

```bash
ruff check .
ruff check . --fix
```

### Flake8 (lint)

```bash
flake8 .
```

### isort (imports)

```bash
isort .
```

---

## 6) Type checking

```bash
mypy .
```

> If your repo ships a `pyproject.toml` or `mypy.ini`, mypy will pick up configuration automatically.

---

## 7) Packaging (build wheels / source dists)

```bash
pip install build
runroot python -m build
```

Artifacts will appear under `dist/` as `*.whl` (wheel) and `*.tar.gz` (sdist).

---

## 8) CLI usage

The tests indicate there is a CLI that responds to `--help` and can render a Mermaid diagram. Depending on how the package is installed, one of these should work:

### a) Installed entry point (replace `accel` with your actual console name if different)

```bash
accel --help
accel diagram --help        # if a diagram subcommand exists
```

### b) runroot python module runner (fallback when unsure of entry point name)

```bash
runroot python -m acceleration.cli --help
runroot python -m acceleration.cli diagram --help   # if applicable
```

> If neither form is recognized, list installed console scripts to discover the exact name:
>
> ```bash
> runroot python -c "import sys,importlib.metadata as m; print('\n'.join(ep.name for ep in m.entry_points().select(group='console_scripts')))"
> ```

---

## 9) Style & quality in one go (if pre-commit is configured)

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

---

## 10) Misc. developer conveniences

### Run a specific subset repeatedly while editing

```bash
ptw acceleration/tests -k euler_quat  # requires pytest-watch
```

### Pin exact tool versions used locally

```bash
pip freeze > requirements-lock.txt
```

---

## 11) Common troubleshooting

- **Wrong runroot python**: Ensure `runroot python --version` shows 3.11.x.
- **vENV not active**: If installing packages fails or tests import the wrong modules, re-activate the venv.
- **Entry point not found**: Try `runroot python -m acceleration.cli --help`.
- **macOS Gatekeeper / permissions**: If scripts aren’t executable, try `chmod +x .venv/bin/*` (Unix) or re-create the venv.
- **NumPy/BLAS mismatch**: If you hit segfaults on Apple Silicon, try `pip install --no-binary=:all: numpy` or use a compatible wheel.

---

Happy hacking! 🚀
