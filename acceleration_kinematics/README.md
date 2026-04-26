# Robotics Project — Run Commands (Full & Exhaustive)

This README lists **ready-to-copy** commands for the most common developer tasks: environment setup, test runs, coverage, formatting, linting, type checking, packaging, and CLI usage. Commands are grouped and include variants (quiet/verbose, single-test, parallel, etc.).

> Target Python: **3.11** (per test logs). Replace `python` with `python3` if your system needs it.

---

## 1) Create & activate a virtual environment

### macOS / Linux (bash / zsh)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
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
pytest acceleration_kinematics/tests
```

### Quiet output (just dots / minimal)

```bash
pytest acceleration_kinematics/tests -q
```

### Verbose output

```bash
pytest acceleration_kinematics/tests -vv
```

### Filter by keyword (only tests with "euler_quat")

```bash
pytest acceleration_kinematics/tests -k euler_quat -vv
```

### Run a single test file

```bash
pytest acceleration_kinematics/tests/test_classic_accel.py -vv
```

### Run a single test (node id)

```bash
pytest acceleration_kinematics/tests/test_classic_accel.py::test_accel_from_euler_derivatives_in_inertial_frame -vv
```

### Stop after first failure

```bash
pytest acceleration_kinematics/tests -x
```

### Re-run last failures only

```bash
pytest acceleration_kinematics/tests --last-failed -vv
```

### Show durations (slowest tests)

```bash
pytest acceleration_kinematics/tests --durations=10 -vv
```

### Run in parallel (auto detect cores)

```bash
pytest acceleration_kinematics/tests -n auto
```

---

## 4) Coverage

### Quick coverage over the whole suite

```bash
pytest acceleration_kinematics/tests --cov=acceleration_kinematics --cov-report=term-missing -vv
```

### Generate HTML coverage report

```bash
pytest acceleration_kinematics/tests --cov=acceleration_kinematics --cov-report=html -vv
# Then open:
python -c "import webbrowser, pathlib; webbrowser.open_new_tab(pathlib.Path('htmlcov/index.html').resolve().as_uri())"
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
python -m build
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

### b) Python module runner (fallback when unsure of entry point name)

```bash
python -m acceleration_kinematics.cli --help
python -m acceleration_kinematics.cli diagram --help   # if applicable
```

> If neither form is recognized, list installed console scripts to discover the exact name:
>
> ```bash
> python -c "import sys,importlib.metadata as m; print('\n'.join(ep.name for ep in m.entry_points().select(group='console_scripts')))"
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
ptw acceleration_kinematics/tests -k euler_quat  # requires pytest-watch
```

### Pin exact tool versions used locally

```bash
pip freeze > requirements-lock.txt
```

---

## 11) Common troubleshooting

- **Wrong Python**: Ensure `python --version` shows 3.11.x.
- **vENV not active**: If installing packages fails or tests import the wrong modules, re-activate the venv.
- **Entry point not found**: Try `python -m acceleration.cli --help`.
- **macOS Gatekeeper / permissions**: If scripts aren’t executable, try `chmod +x .venv/bin/*` (Unix) or re-create the venv.
- **NumPy/BLAS mismatch**: If you hit segfaults on Apple Silicon, try `pip install --no-binary=:all: numpy` or use a compatible wheel.

---

Happy hacking! 🚀
