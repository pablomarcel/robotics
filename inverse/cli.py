# inverse/cli.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import click
import numpy as np

from .apis import InverseService

DEFAULT_IN_DIR = Path("inverse/in")
DEFAULT_OUT_DIR = Path("inverse/out")
_SVC = InverseService()


# ------------------------------- helpers -------------------------------

def _ensure_out_path(path: Optional[Path], default_name: str) -> Path:
    """
    If `path` is None or a directory, return DEFAULT_OUT_DIR/default_name.json.
    If `path` has a suffix, ensure parent exists and return it verbatim.
    """
    if path is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        return DEFAULT_OUT_DIR / f"{default_name}.json"
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{default_name}.json"


def _write_json(obj, out_path: Optional[Path], default_name: str) -> Path:
    p = _ensure_out_path(out_path, default_name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    click.echo(str(p))
    return p


def _float_list(values: Iterable[float]) -> List[float]:
    return list(values)


def _read_json(path: Path):
    if not path.exists():
        raise click.ClickException(f"File not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise click.ClickException(f"Failed to parse JSON: {path} ({exc})") from exc


def _pose_from_cli(x: Optional[float], y: Optional[float], t_path: Optional[Path]):
    if t_path is not None:
        data = _read_json(t_path)
        # Accept either a flat 4x4 list or a nested 4x4 list of lists
        ok_shape = (
            isinstance(data, list)
            and len(data) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in data)
        )
        if not ok_shape:
            raise click.ClickException("--T-path must point to a 4x4 JSON matrix")
        return {"T": data}
    if x is None or y is None:
        raise click.ClickException("Provide either --x/--y or --T-path")
    return {"x": float(x), "y": float(y)}


# -------------------------------- CLI root ------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="inverse", prog_name="inverse-cli")
def cli():
    """Inverse Kinematics CLI — crisp, OOP-friendly, test-driven toolkit."""


# ------------------------------- Commands --------------------------------

@cli.command("problem-validate")
@click.argument("problem_path", type=click.Path(exists=True, path_type=Path))
def cmd_problem_validate(problem_path: Path):
    """Validate an IK problem JSON: {'model':..., 'method':..., 'pose':...}."""
    problem = _read_json(problem_path)
    ok, err = _SVC.validate_problem(problem)
    if ok:
        click.secho("VALID ✓", fg="green")
    else:
        click.secho("INVALID ✗", fg="red")
        raise click.ClickException(err)


@cli.command("problem-solve")
@click.argument("problem_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output file (default inverse/out/solutions.json).")
def cmd_problem_solve(problem_path: Path, out: Optional[Path]):
    """
    Solve an IK problem from file.

    The file must contain:
      {
        "model":  {...},
        "method": {...},
        "pose":   {...}
      }
    """
    problem = _read_json(problem_path)
    ok, err = _SVC.validate_problem(problem)
    if not ok:
        raise click.ClickException(err)
    sols = _SVC.solve(problem["model"], problem["method"], problem["pose"])
    _write_json({"solutions": [np.asarray(s).tolist() for s in sols]}, out, "solutions")


@cli.command("ik-solve")
@click.option("--model", "model_kind", type=click.Choice(["planar2r"]), required=True,
              help="Robot model kind.")
@click.option("--l1", type=float, help="Planar2R: link 1 length.")
@click.option("--l2", type=float, help="Planar2R: link 2 length.")
@click.option("--method", type=click.Choice(["analytic", "iterative"]), default="analytic",
              show_default=True, help="IK method.")
@click.option("--x", type=float, help="Target x (if using x/y pose).")
@click.option("--y", type=float, help="Target y (if using x/y pose).")
@click.option("--T-path", "t_path", type=click.Path(exists=True, path_type=Path),
              help="JSON path to a 4x4 homogeneous matrix for the target pose.")
@click.option("--q0", "q0_vals", type=float, multiple=True,
              help="Initial guess (iterative methods). Repeat per joint.")
@click.option("--tol", type=float, default=1e-6, show_default=True, help="Tolerance (iterative).")
@click.option("--itmax", type=int, default=200, show_default=True, help="Max iters (iterative).")
@click.option("--lambda-damp", type=float, default=1e-3, show_default=True, help="Damping (iterative).")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output file for solutions (default inverse/out/solutions.json).")
def cmd_ik_solve(
    model_kind: str, l1: float, l2: float, method: str,
    x: Optional[float], y: Optional[float], t_path: Optional[Path],
    q0_vals: Iterable[float], tol: float, itmax: int, lambda_damp: float,
    out: Optional[Path],
):
    """Solve IK from command-line parameters (single pose)."""
    # Build model spec
    if model_kind == "planar2r":
        if l1 is None or l2 is None:
            raise click.ClickException("--l1 and --l2 are required for planar2r")
        model_spec = {"kind": "planar2r", "l1": float(l1), "l2": float(l2)}
    else:
        raise click.ClickException(f"Unsupported model: {model_kind}")

    # Method spec
    method_spec = {"method": method}
    if method == "iterative":
        method_spec.update({"tol": tol, "itmax": itmax, "lambda": lambda_damp})

    # Pose spec
    pose_spec = _pose_from_cli(x, y, t_path)

    # Initial guess
    q0 = _float_list(q0_vals) if q0_vals else None

    # Solve
    sols = _SVC.solve(model_spec, method_spec, pose_spec, q0=q0)
    _write_json({"solutions": [np.asarray(s).tolist() for s in sols]}, out, "solutions")


@cli.command("ik-planar2r")
@click.option("--l1", type=float, required=True, help="Link 1 length.")
@click.option("--l2", type=float, required=True, help="Link 2 length.")
@click.option("--x", type=float, required=True, help="Target x.")
@click.option("--y", type=float, required=True, help="Target y.")
@click.option("--method", type=click.Choice(["analytic", "iterative"]), default="analytic",
              show_default=True)
@click.option("--tol", type=float, default=1e-6, show_default=True)
@click.option("--itmax", type=int, default=200, show_default=True)
@click.option("--lambda-damp", type=float, default=1e-3, show_default=True)
@click.option("--q0", "q0_vals", type=float, multiple=True, help="Initial guess for iterative.")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output file (default inverse/out/solutions.json).")
def cmd_ik_planar2r(
    l1: float, l2: float, x: float, y: float, method: str,
    tol: float, itmax: int, lambda_damp: float, q0_vals: Iterable[float],
    out: Optional[Path],
):
    """Convenience 2R planar IK."""
    sols = _SVC.solve_planar2r(
        l1=l1, l2=l2, x=x, y=y, method=method,
        tol=tol, itmax=itmax, lambda_damp=lambda_damp,
        q0=_float_list(q0_vals) if q0_vals else None,
    )
    _write_json({"solutions": [np.asarray(s).tolist() for s in sols]}, out, "solutions")


@cli.command("ik-batch")
@click.option("--model", "model_kind", type=click.Choice(["planar2r"]), required=True)
@click.option("--l1", type=float, help="Planar2R: link 1 length.")
@click.option("--l2", type=float, help="Planar2R: link 2 length.")
@click.option("--poses", "poses_path", type=click.Path(exists=True, path_type=Path), required=True,
              help="JSON file with a list of pose specs (each {'x','y'} or {'T'}).")
@click.option("--method", type=click.Choice(["analytic", "iterative"]), default="analytic", show_default=True)
@click.option("--tol", type=float, default=1e-6, show_default=True)
@click.option("--itmax", type=int, default=200, show_default=True)
@click.option("--lambda-damp", type=float, default=1e-3, show_default=True)
@click.option("--q0", "q0_vals", type=float, multiple=True, help="Initial guess for iterative.")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output file (default inverse/out/batch_solutions.json).")
def cmd_ik_batch(
    model_kind: str, l1: float, l2: float, poses_path: Path, method: str,
    tol: float, itmax: int, lambda_damp: float, q0_vals: Iterable[float],
    out: Optional[Path],
):
    """Batch IK for a list of poses."""
    poses = _read_json(poses_path)
    if not isinstance(poses, list) or not poses:
        raise click.ClickException("--poses must be a JSON list of pose objects")
    if model_kind == "planar2r":
        if l1 is None or l2 is None:
            raise click.ClickException("--l1 and --l2 are required for planar2r")
        model_spec = {"kind": "planar2r", "l1": float(l1), "l2": float(l2)}
    else:
        raise click.ClickException(f"Unsupported model: {model_kind}")

    method_spec = {"method": method}
    if method == "iterative":
        method_spec.update({"tol": tol, "itmax": itmax, "lambda": lambda_damp})
    q0 = _float_list(q0_vals) if q0_vals else None

    results = _SVC.solve_batch(model_spec, method_spec, poses, q0=q0)
    payload = {
        "solutions": [
            [[float(v) for v in sol] for sol in sols_for_pose] for sols_for_pose in results
        ]
    }
    _write_json(payload, out, "batch_solutions")


@cli.command("diagram-mermaid")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output Markdown with Mermaid code (default inverse/out/class_diagram.md).")
def cmd_diagram_mermaid(out: Optional[Path]):
    """Export a Mermaid class diagram of the main OOP types."""
    md = _SVC.class_diagram_mermaid()
    p = _ensure_out_path(out, "class_diagram")
    p.write_text(md, encoding="utf-8")
    click.echo(str(p))


@cli.command("sphinx-skel")
@click.argument("dest", type=click.Path(path_type=Path), default=Path("docs"))
def cmd_sphinx_skel(dest: Path):
    """Create a minimal Sphinx skeleton suitable for API docs."""
    dest.mkdir(parents=True, exist_ok=True)
    conf = (
        '# Generated by inverse.cli\n'
        'project = "inverse"\n'
        'extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]\n'
        'templates_path = ["_templates"]\n'
        'exclude_patterns = []\n'
        'html_theme = "furo"\n'
    )
    index = (
        ".. inverse documentation master file\n\n"
        "Welcome to inverse's docs!\n"
        "==========================\n\n"
        ".. toctree::\n"
        "   :maxdepth: 2\n"
        "   :caption: Contents:\n\n"
        "   api\n"
    )
    api = (
        "API Reference\n"
        "=============\n\n"
        ".. automodule:: inverse.app\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: inverse.apis\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: inverse.core\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: inverse.io\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: inverse.utils\n   :members:\n   :undoc-members:\n"
    )
    makefile = (
        "# Minimal Sphinx Makefile\n"
        ".PHONY: html clean\n"
        "html:\n\t+sphinx-build -b html . _build/html\n"
        "clean:\n\t+rm -rf _build\n"
    )
    (dest / "conf.py").write_text(conf, encoding="utf-8")
    (dest / "index.rst").write_text(index, encoding="utf-8")
    (dest / "api.rst").write_text(api, encoding="utf-8")
    (dest / "Makefile").write_text(makefile, encoding="utf-8")
    click.echo(str(dest))


def main():
    cli(prog_name="inverse-cli")


if __name__ == "__main__":  # pragma: no cover
    main()
