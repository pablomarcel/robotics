
# timeopt/cli.py (upgraded)
"""
Time-optimal control toolbox CLI (click-based)

Improvements ("dirty trick"):
- Robust path handling:
    * --out may be a directory or a file path; we create parents as needed.
    * Defaults keep back-compat with "time/out" but we also accept "timeopt/out".
- Helpful echoes: print where artifacts are written.
- New: `sphinx-skel` command to scaffold minimal Sphinx docs quickly.

Existing commands preserved:
    di         → minimum-time double integrator
    2r-line    → 2R time-opt path scaling along y=const
    diagram    → emit a tiny PlantUML class diagram
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import click

from .apis import DoubleIntegratorAPI, TwoRAPI
from .tools.diagram import emit_puml, write_puml

# ------------------------- Paths & helpers ------------------------------

# Keep legacy default ("time/out") for back-compat with tests/scripts.
LEGACY_DEFAULT_OUT = Path("time/out")
MODERN_DEFAULT_OUT = Path("timeopt/out")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _normalize_out_dir(out: Optional[str]) -> Path:
    """Resolve an output *directory* string, honoring legacy default."""
    if not out:
        # Prefer legacy if it exists, else modern
        if LEGACY_DEFAULT_OUT.exists():
            _ensure_dir(LEGACY_DEFAULT_OUT)
            return LEGACY_DEFAULT_OUT
        _ensure_dir(MODERN_DEFAULT_OUT)
        return MODERN_DEFAULT_OUT
    p = Path(out)
    _ensure_dir(p)
    return p

def _normalize_out_path(out: Optional[str], default_name: str) -> Path:
    """
    Resolve an output *file* path. If `out` is a directory, place default_name inside it.
    If `out` is None, use legacy directory with default_name (fallback to modern).
    """
    if not out:
        base = LEGACY_DEFAULT_OUT if LEGACY_DEFAULT_OUT.exists() else MODERN_DEFAULT_OUT
        _ensure_dir(base)
        return base / default_name
    p = Path(out)
    if p.suffix:
        _ensure_dir(p.parent)
        return p
    # treat as directory
    _ensure_dir(p)
    return p / default_name

# ------------------------------ CLI root --------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    "Time-optimal control toolbox (OOP + TDD)."

# ------------------------------ Commands --------------------------------

@cli.command("di")
@click.option("--x0", type=float, required=True, help="Initial position")
@click.option("--xf", type=float, required=True, help="Final position")
@click.option("--m", type=float, default=1.0, show_default=True, help="Mass")
@click.option("--F", "Fmax", type=float, default=10.0, show_default=True, help="Max force |F| ≤ Fmax")
@click.option("--mu", type=float, default=0.0, show_default=True, help="Coulomb friction coef.")
@click.option("--drag", type=float, default=0.0, show_default=True, help="Linear drag coef.")
@click.option("--out", type=str, default=str(LEGACY_DEFAULT_OUT), show_default=True,
              help="Output directory (legacy default 'time/out')")
def run_di(x0, xf, m, Fmax, mu, drag, out):
    """Minimum-time double-integrator (CasADi)."""
    out_dir = _normalize_out_dir(out)
    res = DoubleIntegratorAPI().solve(x0=x0, xf=xf, m=m, F=Fmax, mu=mu, drag=drag, out_dir=str(out_dir))
    # API is assumed to write artifacts inside out_dir; echo a summary/path.
    click.echo(res.data if hasattr(res, "data") else f"Wrote artifacts to {out_dir}")

@cli.command("2r-line")
@click.option("--y", type=float, required=True, help="Constant y for the line path")
@click.option("--x0", type=float, required=True, help="Start x")
@click.option("--x1", type=float, required=True, help="End x")
@click.option("--n", type=int, default=200, show_default=True, help="Number of samples")
@click.option("--tau", type=str, default="100,100", show_default=True, help="Tau limits as 'Pmax,Qmax'")
@click.option("--out", type=str, default=str(LEGACY_DEFAULT_OUT), show_default=True,
              help="Output directory (legacy default 'time/out')")
def run_2r_line(y, x0, x1, n, tau, out):
    """Time-optimal along line y=const with TOPPRA (2R)."""
    try:
        tmax = tuple(float(s) for s in str(tau).split(","))
        if len(tmax) != 2:
            raise ValueError
    except Exception:
        raise click.ClickException("--tau must be two comma-separated numbers, e.g., '120,100'")
    out_dir = _normalize_out_dir(out)
    res = TwoRAPI().line_y(y=y, x0=x0, x1=x1, N=n, tau_max=tmax, out_dir=str(out_dir))
    click.echo(res.data if hasattr(res, "data") else f"Wrote artifacts to {out_dir}")

@cli.command("diagram")
@click.option("--out", type=str, default=str(LEGACY_DEFAULT_OUT / "time.puml"),
              show_default=True, help="Output .puml path")
def diagram(out):
    """Emit a minimal PlantUML class diagram."""
    classes = ["TimeOptimalProblem", "MinTimeDoubleIntegrator", "TwoRPathTimeScaler", "Planar2RGeom"]
    edges = [("MinTimeDoubleIntegrator","TimeOptimalProblem"),
             ("TwoRPathTimeScaler","TimeOptimalProblem")]
    content = emit_puml(classes, edges)
    path = _normalize_out_path(out, "time.puml")
    p = write_puml(str(path), content)
    click.echo(f"Wrote {p}")

# --------------------------- Sphinx skeleton ----------------------------

@cli.command("sphinx-skel")
@click.argument("dest", required=False, default="docs")
def sphinx_skel(dest: str):
    """Create a minimal Sphinx docs skeleton (like other modules)."""
    dest_path = Path(dest)
    _ensure_dir(dest_path)
    conf = (
        '# Generated by timeopt.cli\n'
        'project = "timeopt"\n'
        'extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]\n'
        'templates_path = ["_templates"]\n'
        'exclude_patterns = []\n'
        'html_theme = "furo"\n'
    )
    index = (
        ".. timeopt documentation master file\n\n"
        "Welcome to timeopt's docs!\n"
        "==========================\n\n"
        ".. toctree::\n"
        "   :maxdepth: 2\n"
        "   :caption: Contents:\n\n"
        "   api\n"
    )
    api = (
        "API Reference\n"
        "=============\n\n"
        ".. automodule:: timeopt.apis\n   :members:\n   :undoc-members:\n\n"
        ".. automodule:: timeopt.tools.diagram\n   :members:\n   :undoc-members:\n\n"
    )
    makefile = (
        "# Minimal Sphinx Makefile\n"
        ".PHONY: html clean\n"
        "html:\n\t+sphinx-build -b html . _build/html\n"
        "clean:\n\t+rm -rf _build\n"
    )
    def _w(p: Path, txt: str):
        if not p.exists():
            p.write_text(txt, encoding="utf-8")
    (dest_path / "_templates").mkdir(exist_ok=True)
    (dest_path / "_static").mkdir(exist_ok=True)
    _w(dest_path / "conf.py", conf)
    _w(dest_path / "index.rst", index)
    _w(dest_path / "api.rst", api)
    _w(dest_path / "Makefile", makefile)
    click.echo(str(dest_path))

if __name__ == "__main__":  # pragma: no cover
    cli()
