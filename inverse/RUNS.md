# inverse — Run Commands Cheat‑Sheet

This README captures a **full and exhaustive** list of practical run commands
that mirror and extend what the automated test-suite exercises. Everything
assumes you are at the repository root with your virtualenv active.

> Replace paths as needed for your environment (macOS/Linux shown).

---

## 0) Run the full test suite

```bash
pytest inverse/tests -q
```

---

## 1) Quick single‑pose solves (2R planar)

These use the CLI’s two entrypoints: `ik-planar2r` (convenience wrapper) and the
more general `ik-solve` which accepts model/method flags.

### 1.1 Analytic — using x/y directly

```bash
inverse-cli ik-planar2r --l1 1.0 --l2 1.0 --x 1.2 --y 0.3   --method analytic   -o inverse/out/solutions_analytic_planar2r_xy.json
```

### 1.2 Iterative — using x/y directly (with a seed)

```bash
inverse-cli ik-planar2r --l1 1.0 --l2 1.0 --x 0.5 --y 1.0   --method iterative --tol 1e-6 --itmax 200 --lambda-damp 1e-3   --q0 0.0 --q0 0.0   -o inverse/out/solutions_iterative_planar2r_xy.json
```

### 1.3 Analytic via the generic solver — using x/y

```bash
inverse-cli ik-solve --model planar2r --l1 1.0 --l2 1.0   --method analytic --x 1.2 --y 0.3   -o inverse/out/solutions_analytic_generic_xy.json
```

### 1.4 Iterative via the generic solver — using x/y

```bash
inverse-cli ik-solve --model planar2r --l1 1.0 --l2 1.0   --method iterative --x 0.0 --y 1.8   --q0 0.1 --q0 -0.1 --tol 1e-8 --itmax 200 --lambda-damp 1e-3   -o inverse/out/solutions_iterative_generic_xy.json
```

### 1.5 Iterative via the generic solver — using a 4×4 target transform (file)

Create a target pose file (pure translation to x=1.2, y=0.3):

```bash
cat > /tmp/target_T.json << 'JSON'
[[1,0,0,1.2],
 [0,1,0,0.3],
 [0,0,1,0.0],
 [0,0,0,1.0]]
JSON
```

Solve iteratively from that file:

```bash
inverse-cli ik-solve --model planar2r --l1 1.0 --l2 1.0   --method iterative --T-path /tmp/target_T.json   --q0 0.1 --q0 0.1 --tol 1e-8 --itmax 200 --lambda-damp 1e-3   -o inverse/out/solutions_iterative_generic_T.json
```

> The tests specifically use this variant.

---

## 2) Batch solve (multiple poses)

Prepare a batch JSON with mixed pose styles (x/y and 4×4 `T`):

```bash
cat > /tmp/poses.json << 'JSON'
[
  {"x": 1.2, "y": 0.3},
  {"T": [[1,0,0,0.5],[0,1,0,1.0],[0,0,1,0.0],[0,0,0,1]]},
  {"x": 0.0, "y": 1.8}
]
JSON
```

### 2.1 Analytic batch

```bash
inverse-cli ik-batch --model planar2r --l1 1.0 --l2 1.0   --poses /tmp/poses.json --method analytic   -o inverse/out/batch_solutions_analytic.json
```

### 2.2 Iterative batch (with seed and DLS params)

```bash
inverse-cli ik-batch --model planar2r --l1 1.0 --l2 1.0   --poses /tmp/poses.json --method iterative   --tol 1e-6 --itmax 200 --lambda-damp 1e-3 --q0 0.0 --q0 0.0   -o inverse/out/batch_solutions_iterative.json
```

---

## 3) Problem file: validate and solve

### 3.1 Create a valid problem file

```bash
cat > /tmp/problem_ok.json << 'JSON'
{
  "model":  {"kind": "planar2r", "l1": 1.0, "l2": 1.0},
  "method": {"method": "iterative", "tol": 1e-9, "itmax": 250, "lambda": 1e-3},
  "pose":   {"T": [[1,0,0,1.2],[0,1,0,0.3],[0,0,1,0.0],[0,0,0,1.0]]}
}
JSON
```

### 3.2 Validate

```bash
inverse-cli problem-validate /tmp/problem_ok.json
```

### 3.3 Solve

```bash
inverse-cli problem-solve /tmp/problem_ok.json   -o inverse/out/solutions_problem.json
```

---

## 4) Docs & diagrams

### 4.1 Mermaid class diagram (Markdown with Mermaid block)

```bash
inverse-cli diagram-mermaid -o inverse/out/class_diagram.md
```

### 4.2 Minimal Sphinx skeleton (then optional HTML build)

```bash
inverse-cli sphinx-skel docs
make -C docs html
```

---

## 5) Helpful introspection

(Not required by tests, but useful.)

```bash
inverse-cli --help
inverse-cli ik-solve --help
inverse-cli ik-planar2r --help
inverse-cli ik-batch --help
inverse-cli problem-validate --help
inverse-cli problem-solve --help
inverse-cli diagram-mermaid --help
inverse-cli sphinx-skel --help
```

---

## Notes

- All CLI outputs are JSON unless stated otherwise.
- When passing multiple `--q0` values, repeat the flag once per joint.
- Iterative solver uses DLS: `Δq = (JᵀJ + λ²I)⁻¹Jᵀe` with error stacked as `[ω; dp]`
  to match the Jacobian’s `[ω; v]` column convention.
