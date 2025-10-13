
# inverse — Run Commands Cheat‑Sheet (clean)

This doc lists **practical, test-backed run commands**. Assumes you're at any
path inside the repo, with your virtualenv active. The helper below ensures
commands always execute from the project root.

---

## 0) One‑time shell helper (define once per shell)

```bash
# Project-root helper (paste once per shell)
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
runroot mkdir -p inverse/out
```

---

## 1) Run the full test suite

```bash
runroot python -m pytest inverse/tests -q
```

---

## 2) Quick single‑pose solves (2R planar)

### 2.1 Analytic — convenience command (x/y)

```bash
runroot python -m inverse.cli ik-planar2r --l1 1.0 --l2 1.0 --x 1.2 --y 0.3 --method analytic -o inverse/out/solutions_analytic_planar2r_xy.json
```

### 2.2 Iterative — convenience command (x/y, with seed)

```bash
runroot python -m inverse.cli ik-planar2r --l1 1.0 --l2 1.0 --x 0.5 --y 1.0 --method iterative --tol 1e-6 --itmax 200 --lambda-damp 1e-3 --q0 0.0 --q0 0.0 -o inverse/out/solutions_iterative_planar2r_xy.json
```

### 2.3 Analytic — generic solver (x/y)

```bash
runroot python -m inverse.cli ik-solve --model planar2r --l1 1.0 --l2 1.0 --method analytic --x 1.2 --y 0.3 -o inverse/out/solutions_analytic_generic_xy.json
```

### 2.4 Iterative — generic solver (x/y)

```bash
runroot python -m inverse.cli ik-solve --model planar2r --l1 1.0 --l2 1.0 --method iterative --x 0.0 --y 1.8 --q0 0.1 --q0 -0.1 --tol 1e-8 --itmax 200 --lambda-damp 1e-3 -o inverse/out/solutions_iterative_generic_xy.json
```

### 2.5 Iterative — generic solver using a 4×4 target transform (file)

Create target pose file:

```bash
cat > /tmp/target_T.json << 'JSON'
[[1,0,0,1.2],
 [0,1,0,0.3],
 [0,0,1,0.0],
 [0,0,0,1.0]]
JSON
```

Solve:

```bash
runroot python -m inverse.cli ik-solve --model planar2r --l1 1.0 --l2 1.0 --method iterative --T-path inverse/in/target_T.json --q0 0.1 --q0 0.1 --tol 1e-8 --itmax 200 --lambda-damp 1e-3 -o inverse/out/solutions_iterative_generic_T.json
```

---

## 3) Batch solve (multiple poses)

Prepare batch file:

```bash
cat > /tmp/poses.json << 'JSON'
[
  {"x": 1.2, "y": 0.3},
  {"T": [[1,0,0,0.5],[0,1,0,1.0],[0,0,1,0.0],[0,0,0,1]]},
  {"x": 0.0, "y": 1.8}
]
JSON
```

### 3.1 Analytic batch

```bash
runroot python -m inverse.cli ik-batch --model planar2r --l1 1.0 --l2 1.0 --poses inverse/in/poses.json --method analytic -o inverse/out/batch_solutions_analytic.json
```

### 3.2 Iterative batch (with seed & DLS params)

```bash
runroot python -m inverse.cli ik-batch --model planar2r --l1 1.0 --l2 1.0 --poses inverse/in/poses.json --method iterative --tol 1e-6 --itmax 200 --lambda-damp 1e-3 --q0 0.0 --q0 0.0 -o inverse/out/batch_solutions_iterative.json
```

---

## 4) Problem files: validate & solve

Create a problem file:

```bash
cat > /tmp/problem_ok.json << 'JSON'
{
  "model":  {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
  "method": {"method": "iterative", "tol": 1e-9, "itmax": 250, "lambda": 1e-3},
  "pose":   {"T": [[1,0,0,1.2],[0,1,0,0.3],[0,0,1,0.0],[0,0,0,1.0]]}
}
JSON
```

Validate:

```bash
runroot python -m inverse.cli problem-validate inverse/in/problem_ok.json
```

Solve:

```bash
runroot python -m inverse.cli problem-solve inverse/in/problem_ok.json -o inverse/out/solutions_problem.json
```

---

## 5) Docs & diagrams

Mermaid class diagram (Markdown):

```bash
runroot python -m inverse.cli diagram-mermaid -o inverse/out/class_diagram.md
```

Minimal Sphinx skeleton (optionally build HTML):

```bash
runroot python -m inverse.cli sphinx-skel docs
```

```bash
make -C docs html
```

---

## 6) CLI help

```bash
runroot python -m inverse.cli --help
```

```bash
runroot python -m inverse.cli ik-solve --help
```

```bash
runroot python -m inverse.cli ik-planar2r --help
```

```bash
runroot python -m inverse.cli ik-batch --help
```

```bash
runroot python -m inverse.cli problem-validate --help
```

```bash
runroot python -m inverse.cli problem-solve --help
```

```bash
runroot python -m inverse.cli diagram-mermaid --help
```

```bash
runroot python -m inverse.cli sphinx-skel --help
```

---

### Notes
- JSON is the default output format for solver commands.
- For iterative IK, the DLS update is \(\Delta q=(J^T J + \lambda^2 I)^{-1} J^T e\) with error stacked as `[ω; dp]` to match the Jacobian’s `[ω; v]` columns.
