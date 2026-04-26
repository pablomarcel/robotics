# Applied Dynamics — Clean Run Commands

This sheet lists **only** the project‑root (`runroot`) **Python** commands you’ll typically use, plus a one‑time bootstrap snippet.

> All commands assume you paste the bootstrap into your shell first. Each command below is in its own code block for copy/paste.

---

## -1) One‑time shell bootstrap (paste once per new shell)

```bash
# Find project root via Git or common markers
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

# Run any command from the repo root in a subshell
runroot() { ( cd "$(_mc_root)" && "$@" ); }
```

```bash
runroot python -c "import pathlib; (pathlib.Path('applied')/'out').mkdir(parents=True, exist_ok=True)"
```

---

## 0) Environment setup

```bash
runroot python -m venv .venv
```

```bash
runroot python -m pip install -e .
```

```bash
runroot python -m pip install graphviz
```

---

## 1) CLI — Design presets

```bash
runroot python -m applied_dynamics.cli design --list
```

```bash
runroot python -m applied_dynamics.cli design --preset pendulum_sym
```

```bash
runroot python -m applied_dynamics.cli design --preset spherical_sym
```

```bash
runroot python -m applied_dynamics.cli design --preset planar2r_sym
```

```bash
runroot python -m applied_dynamics.cli design --preset absorber_sym
```

```bash
runroot python -m applied_dynamics.cli design --preset pendulum_num
```

```bash
runroot python -m applied_dynamics.cli design --preset planar2r_num
```

```bash
runroot python -m applied_dynamics.cli design --preset absorber_num
```

```bash
runroot python -m applied_dynamics.cli design --preset planar2r_num --export applied_dynamics/out/planar2r_num.json
```

---

## 2) CLI — Diagram tooling

```bash
runroot python -m applied_dynamics.cli diagram dot --out applied_dynamics/out/classes.dot
```

```bash
runroot python -m applied_dynamics.cli diagram dot --packages applied_dynamics.core,applied_dynamics.models --outdir applied_dynamics/out --theme dark --rankdir TB --legend --no-cluster --out applied_dynamics/out/classes_custom.dot
```

```bash
runroot python -m applied_dynamics.cli diagram plantuml --out applied_dynamics/out/classes.puml
```

```bash
runroot python -m applied_dynamics.cli diagram json --out applied_dynamics/out/classes.json
```

```bash
runroot python -m applied_dynamics.cli diagram graphviz --fmt png --dpi 220 --outstem applied_dynamics/out/classes
```

```bash
runroot python -m applied_dynamics.cli diagram graphviz --fmt svg --outstem applied_dynamics/out/classes_svg
```

```bash
runroot python -m applied_dynamics.cli diagram graphviz --fmt pdf --dpi 260 --rankdir TB --theme dark --outstem applied_dynamics/out/classes_pdf
```

```bash
runroot python -m applied_dynamics.cli diagram all
```

```bash
runroot python -m applied_dynamics.cli diagram all --packages applied_dynamics.core,applied_dynamics.models --outdir applied_dynamics/out --theme dark --rankdir TB --legend
```

---

## 3) CLI — Preset aliases

```bash
runroot python -m applied_dynamics.cli pendulum
```

```bash
runroot python -m applied_dynamics.cli spherical
```

```bash
runroot python -m applied_dynamics.cli planar2r
```

```bash
runroot python -m applied_dynamics.cli absorber
```

---

## 4) Testing (pytest via Python module)

```bash
runroot python -m pytest -q
```

```bash
runroot python -m pytest applied_dynamics/tests -q
```

---

## 5) Outputs

Most commands write to `applied/out/`. Adjust paths as needed.
