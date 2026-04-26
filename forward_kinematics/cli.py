# forward_kinematics/cli.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import click
import numpy as np

from .apis import ForwardService
from . import io as io_mod

DEFAULT_IN_DIR = Path("forward_kinematics/in")
DEFAULT_OUT_DIR = Path("forward_kinematics/out")
_SVC = ForwardService()


def _ensure_out_dir(path: Optional[Path]) -> Path:
    if path is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        return DEFAULT_OUT_DIR
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_spec(path: Path) -> dict:
    if not path.exists():
        raise click.ClickException(f"Spec not found: {path}")
    return io_mod.load_spec_from_file(str(path))


def _write_ndarray(arr: np.ndarray, out: Optional[Path], basename: str):
    if out is None or not out.suffix:
        final = DEFAULT_OUT_DIR / f"{basename}.json"
        final.parent.mkdir(parents=True, exist_ok=True)
        with open(final, "w", encoding="utf-8") as f:
            json.dump(arr.tolist(), f, indent=2)
        click.echo(str(final))
        return
    out = _ensure_out_dir(out)
    if out.suffix.lower() == ".npy":
        np.save(out, arr)
    elif out.suffix.lower() in {".txt", ".csv"}:
        np.savetxt(out, arr, delimiter="," if out.suffix.lower() == ".csv" else " ")
    else:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(arr.tolist(), f, indent=2)
    click.echo(str(out))


def _write_transform(T, out: Optional[Path]):
    _write_ndarray(T.as_matrix(), out, "transform")


def _float_list(values: Iterable[float]) -> List[float]:
    # now values are already floats (click handles conversion)
    return list(values)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="forward_kinematics", prog_name="forward_kinematics-cli")
def cli():
    """Forward Kinematics CLI — beautiful, crisp, OOP-friendly toolkit."""


@cli.command("validate")
@click.argument("spec_path", type=click.Path(exists=True, path_type=Path))
def cmd_validate(spec_path: Path):
    """Validate a JSON/YAML robot specification against the JSON Schema."""
    spec = _read_spec(spec_path)
    ok, err = _SVC.validate(spec)
    if ok:
        click.secho("VALID ✓", fg="green")
    else:
        click.secho("INVALID ✗", fg="red")
        raise click.ClickException(err)


@cli.command("schema")
@click.option("-o", "--out", type=click.Path(path_type=Path), help="Output path (.json).")
def cmd_schema(out: Optional[Path]):
    """Export the JSON Schema used by the toolkit."""
    schema = _SVC.schema()
    out = out or (DEFAULT_OUT_DIR / "robot.schema.json")
    out = _ensure_out_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    click.echo(str(out))


@cli.command("fk")
@click.argument("spec_path", type=click.Path(exists=True, path_type=Path))
@click.option("--q", "q_vals", type=float, multiple=True, required=True,
              help="Repeat per joint: --q q1 --q q2 ...")
@click.option("-o", "--out", type=click.Path(path_type=Path), help="Output file for 4x4 T.")
def cmd_fk(spec_path: Path, q_vals: Iterable[float], out: Optional[Path]):
    """Compute forward_kinematics kinematics and write a 4x4 homogeneous transform."""
    spec = _read_spec(spec_path)
    ok, err = _SVC.validate(spec)
    if not ok:
        raise click.ClickException(err)
    chain = _SVC.load_spec(spec, validate=False)
    T = _SVC.forward_kinematics(chain, _float_list(q_vals))
    _write_transform(T, out)


@cli.command("jacobian-space")
@click.argument("spec_path", type=click.Path(exists=True, path_type=Path))
@click.option("--q", "q_vals", type=float, multiple=True, required=True,
              help="Repeat per joint: --q q1 --q q2 ...")
@click.option("-o", "--out", type=click.Path(path_type=Path), help="Output file for J_space.")
def cmd_jac_space(spec_path: Path, q_vals: Iterable[float], out: Optional[Path]):
    """Compute the analytical space Jacobian J_s(q)."""
    spec = _read_spec(spec_path)
    ok, err = _SVC.validate(spec)
    if not ok:
        raise click.ClickException(err)
    chain = _SVC.load_spec(spec, validate=False)
    J = _SVC.jacobian_space(chain, _float_list(q_vals))
    _write_ndarray(J, out, "jacobian_space")


@cli.command("jacobian-body")
@click.argument("spec_path", type=click.Path(exists=True, path_type=Path))
@click.option("--q", "q_vals", type=float, multiple=True, required=True,
              help="Repeat per joint: --q q1 --q q2 ...")
@click.option("-o", "--out", type=click.Path(path_type=Path), help="Output file for J_body.")
def cmd_jac_body(spec_path: Path, q_vals: Iterable[float], out: Optional[Path]):
    """Compute the analytical body Jacobian J_b(q)."""
    spec = _read_spec(spec_path)
    ok, err = _SVC.validate(spec)
    if not ok:
        raise click.ClickException(err)
    chain = _SVC.load_spec(spec, validate=False)
    J = _SVC.jacobian_body(chain, _float_list(q_vals))
    _write_ndarray(J, out, "jacobian_body")


@cli.command("preset-scara")
@click.option("--l1", type=float, required=True, help="Link 1 length.")
@click.option("--l2", type=float, required=True, help="Link 2 length.")
@click.option("--d", type=float, default=0.0, show_default=True, help="Vertical offset.")
@click.option("--q", "q_vals", type=float, multiple=True, required=True,
              help="SCARA joint values (repeat): --q q1 --q q2 --q d3 [--q q4]")
@click.option("-o", "--out", type=click.Path(path_type=Path), help="Output T (default JSON).")
def cmd_preset_scara(l1: float, l2: float, d: float, q_vals: Iterable[float], out: Optional[Path]):
    """Run FK + space Jacobian on a SCARA preset."""
    chain = _SVC.preset_scara(l1, l2, d)
    qv = _float_list(q_vals)
    T = _SVC.forward_kinematics(chain, qv)
    J = _SVC.jacobian_space(chain, qv)
    _write_transform(T, out)
    stem = (out if out and out.suffix else DEFAULT_OUT_DIR / "transform.json")
    jac_out = (stem.parent / (stem.stem + "_J_space.json"))
    _write_ndarray(J, jac_out, "jacobian_space")


@cli.command("preset-wrist")
@click.option("--type", "wrist_type", type=click.IntRange(1, 3), required=True, help="Wrist type 1–3.")
@click.option("--d7", type=float, default=0.0, show_default=True, help="Tool offset (along z7).")
@click.option("--q", "q_vals", type=float, multiple=True, required=True,
              help="Wrist joint values (repeat): --q q4 --q q5 --q q6")
@click.option("-o", "--out", type=click.Path(path_type=Path), help="Output T (default JSON).")
def cmd_preset_wrist(wrist_type: int, d7: float, q_vals: Iterable[float], out: Optional[Path]):
    """Run FK + body Jacobian on a spherical wrist preset (types 1–3)."""
    chain = _SVC.preset_spherical_wrist(wrist_type=wrist_type, d7=d7)
    qv = _float_list(q_vals)
    T = _SVC.forward_kinematics(chain, qv)
    Jb = _SVC.jacobian_body(chain, qv)
    _write_transform(T, out)
    stem = (out if out and out.suffix else DEFAULT_OUT_DIR / "transform.json")
    jac_out = (stem.parent / (stem.stem + "_J_body.json"))
    _write_ndarray(Jb, jac_out, "jacobian_body")


@cli.command("diagram-dot")
@click.option("-o", "--out", type=click.Path(path_type=Path), help="Output DOT file.")
def cmd_diagram_dot(out: Optional[Path]):
    """Export a Graphviz DOT class diagram of the main OOP types."""
    from .tools.diagram import render_dot  # local import to keep CLI snappy
    dot = render_dot()
    out = out or (DEFAULT_OUT_DIR / "classes.dot")
    out = _ensure_out_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(dot)
    click.echo(str(out))


@cli.command("sphinx-skel")
@click.argument("dest", type=click.Path(path_type=Path), default=Path("docs"))
def cmd_sphinx_skel(dest: Path):
    """Create a minimal Sphinx skeleton suitable for API docs."""
    dest.mkdir(parents=True, exist_ok=True)
    conf = (
        '# Generated by forward_kinematics.cli\n'
        'project = "forward_kinematics"\n'
        'extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]\n'
        'templates_path = ["_templates"]\n'
        'exclude_patterns = []\n'
        'html_theme = "furo"\n'
    )
    index = (
        ".. forward_kinematics documentation master file\n\n"
        "Welcome to forward_kinematics's docs!\n"
        "==========================\n\n"
        ".. toctree::\n"
        "   :maxdepth: 2\n"
        "   :caption: Contents:\n\n"
        "   api\n"
    )
    api = (
        "API Reference\n"
        "=============\n\n"
        ".. automodule:: forward_kinematics.app\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: forward_kinematics.core\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: forward_kinematics.design\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: forward_kinematics.io\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: forward_kinematics.utils\n   :members:\n   :undoc-members:\n"
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
    cli(prog_name="forward_kinematics-cli")


if __name__ == "__main__":  # pragma: no cover
    main()
