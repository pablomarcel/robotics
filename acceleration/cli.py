# acceleration/cli.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import click
import numpy as np

from .apis import AccelService

DEFAULT_IN_DIR = Path("acceleration/in")
DEFAULT_OUT_DIR = Path("acceleration/out")
_SVC = AccelService()


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
    return [float(v) for v in values]


def _read_json(path: Path):
    if not path.exists():
        raise click.ClickException(f"File not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise click.ClickException(f"Failed to parse JSON: {path} ({exc})") from exc


def _read_matrix_json(path: Path, *, shape: tuple[int, int]) -> List[List[float]]:
    data = _read_json(path)
    r, c = shape
    ok = isinstance(data, list) and len(data) == r and all(isinstance(row, list) and len(row) == c for row in data)
    if not ok:
        raise click.ClickException(f"JSON at {path} must be a {r}x{c} list-of-lists")
    return data


# -------------------------------- CLI root ------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="acceleration", prog_name="accel-cli")
def cli():
    """Acceleration Kinematics CLI — OOP, test-driven, and crisp."""


# ------------------------------- Problem I/O -----------------------------

@cli.command("problem-validate")
@click.argument("problem_path", type=click.Path(exists=True, path_type=Path))
def cmd_problem_validate(problem_path: Path):
    """Validate a generic acceleration problem JSON: {'op':..., 'payload':..., 'model'?:...}."""
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
              help="Output file (default acceleration/out/result.json).")
def cmd_problem_solve(problem_path: Path, out: Optional[Path]):
    """
    Solve a generic acceleration problem from file.

    The file must contain:
      {
        "op": "...",
        "payload": {...},
        "model": {...}   # only required for forward_kinematics/inverse_kinematics (e.g., {'kind':'planar2r','l1':1,'l2':1})
      }
    """
    problem = _read_json(problem_path)
    ok, err = _SVC.validate_problem(problem)
    if not ok:
        raise click.ClickException(err)
    result = _SVC.solve(problem)
    _write_json({"result": result}, out, "result")


# ------------------------------ Core commands ---------------------------

@cli.command("forward_kinematics")
@click.option("--l1", type=float, required=True, help="Planar2R: link 1 length.")
@click.option("--l2", type=float, required=True, help="Planar2R: link 2 length.")
@click.option("--q", type=float, multiple=True, required=True, help="Joint positions (repeat per joint).")
@click.option("--qd", type=float, multiple=True, required=True, help="Joint velocities (repeat per joint).")
@click.option("--qdd", type=float, multiple=True, required=True, help="Joint accelerations (repeat per joint).")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output file (default acceleration/out/xdd.json).")
def cmd_forward(l1: float, l2: float, q: Iterable[float], qd: Iterable[float], qdd: Iterable[float],
                out: Optional[Path]):
    """Forward acceleration (e.g., 2R planar end-effector Ẍ = J q̈ + Ĵ q̇)."""
    xdd = _SVC.forward(l1, l2, _float_list(q), _float_list(qd), _float_list(qdd))
    _write_json({"xdd": np.asarray(xdd).tolist()}, out, "xdd")


@cli.command("inverse_kinematics")
@click.option("--l1", type=float, required=True, help="Planar2R: link 1 length.")
@click.option("--l2", type=float, required=True, help="Planar2R: link 2 length.")
@click.option("--q", type=float, multiple=True, required=True, help="Joint positions.")
@click.option("--qd", type=float, multiple=True, required=True, help="Joint velocities.")
@click.option("--xdd", type=float, multiple=True, required=True, help="Target end-effector acceleration (2 or 3 comps).")
@click.option("--damping", type=float, default=1e-8, show_default=True, help="DLS damping for J⁻¹.")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output file (default acceleration/out/qdd.json).")
def cmd_inverse(l1: float, l2: float, q: Iterable[float], qd: Iterable[float], xdd: Iterable[float],
                damping: float, out: Optional[Path]):
    """Inverse acceleration (solve for q̈ given Ẍ)."""
    qdd = _SVC.inverse(l1, l2, _float_list(q), _float_list(qd), _float_list(xdd), damping=damping)
    _write_json({"qdd": np.asarray(qdd).tolist()}, out, "qdd")


@cli.command("classic")
@click.option("--alpha", type=float, multiple=True, required=True, help="Angular accel α (3 comps).")
@click.option("--omega", type=float, multiple=True, required=True, help="Angular vel ω (3 comps).")
@click.option("--r", type=float, multiple=True, required=True, help="Position vector r (3 comps).")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output file (default acceleration/out/a_classic.json).")
def cmd_classic(alpha: Iterable[float], omega: Iterable[float], r: Iterable[float], out: Optional[Path]):
    """Classic rigid-body acceleration: a = α×r + ω×(ω×r)."""
    a = _SVC.classic(_float_list(alpha), _float_list(omega), _float_list(r))
    _write_json({"a": np.asarray(a).tolist()}, out, "a_classic")


@cli.command("euler-alpha")
@click.option("--angles", type=float, multiple=True, required=True, help="ZYX Euler angles (3).")
@click.option("--rates", type=float, multiple=True, required=True, help="Angle rates (3).")
@click.option("--accels", type=float, multiple=True, required=True, help="Angle accelerations (3).")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output file (default acceleration/out/alpha_euler.json).")
def cmd_euler_alpha(angles: Iterable[float], rates: Iterable[float], accels: Iterable[float], out: Optional[Path]):
    """Compute angular_velocity acceleration α from ZYX Euler angles/ rates/ accelerations."""
    alpha = _SVC.euler_alpha(_float_list(angles), _float_list(rates), _float_list(accels))
    _write_json({"alpha": np.asarray(alpha).tolist()}, out, "alpha_euler")


@cli.command("quat-sb")
@click.option("--q", type=float, multiple=True, required=True, help="Quaternion (w x y z).")
@click.option("--qd", type=float, multiple=True, required=True, help="Quaternion rate.")
@click.option("--qdd", type=float, multiple=True, required=True, help="Quaternion acceleration.")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output file (default acceleration/out/S_B.json).")
def cmd_quat_sb(q: Iterable[float], qd: Iterable[float], qdd: Iterable[float], out: Optional[Path]):
    """
    Compute B-frame 'S' matrix (α~ + ω~^2) from quaternion kinematics
    — useful for (9.175–9.181) style formulations.
    """
    S = _SVC.quat_sb(_float_list(q), _float_list(qd), _float_list(qdd))
    _write_json({"S_B": np.asarray(S).tolist()}, out, "S_B")


@cli.command("mixed")
@click.option("--R-path", "r_path", type=click.Path(exists=True, path_type=Path), required=True,
              help="JSON path to a 3x3 rotation_kinematics matrix.")
@click.option("--omega", type=float, multiple=True, required=True, help="ω of B in G (3).")
@click.option("--alpha", type=float, multiple=True, required=True, help="α of B in G (3).")
@click.option("--r", type=float, multiple=True, required=True, help="Position vector r (3).")
@click.option("--vB", type=float, multiple=True, required=True, help="Velocity of the body point in B (3).")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output file (default acceleration/out/mixed.json).")
def cmd_mixed(r_path: Path, omega: Iterable[float], alpha: Iterable[float],
              r: Iterable[float], vB: Iterable[float], out: Optional[Path]):
    """
    Mixed accelerations (9.400–9.426 family): compute B-expression of G-accel
    and G-expression of B-accel for a moving point; thin wrapper over service.
    """
    R = _read_matrix_json(r_path, shape=(3, 3))
    res = _SVC.mixed(R, _float_list(omega), _float_list(alpha), _float_list(r), _float_list(vB))
    _write_json(res, out, "mixed")


# --------------------------- Diagram & Docs -----------------------------

@cli.command("diagram-mermaid")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Output Markdown with Mermaid code (default acceleration/out/class_diagram.md).")
def cmd_diagram_mermaid(out: Optional[Path]):
    """Export a Mermaid class diagram of the main OOP types in acceleration module."""
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
        '# Generated by acceleration.cli\n'
        'project = "acceleration"\n'
        'extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]\n'
        'templates_path = ["_templates"]\n'
        'exclude_patterns = []\n'
        'html_theme = "furo"\n'
    )
    index = (
        ".. acceleration documentation master file\n\n"
        "Welcome to acceleration's docs!\n"
        "================================\n\n"
        ".. toctree::\n"
        "   :maxdepth: 2\n"
        "   :caption: Contents:\n\n"
        "   api\n"
    )
    api = (
        "API Reference\n"
        "=============\n\n"
        ".. automodule:: acceleration.app\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: acceleration.apis\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: acceleration.core\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: acceleration.io\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: acceleration.utils\n   :members:\n   :undoc-members:\n"
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
    cli(prog_name="accel-cli")


if __name__ == "__main__":  # pragma: no cover
    main()
