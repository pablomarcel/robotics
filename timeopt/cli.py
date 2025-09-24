# time/cli.py
from __future__ import annotations
import click
from .apis import DoubleIntegratorAPI, TwoRAPI
from .tools.diagram import emit_puml, write_puml

@click.group()
def cli():
    "Time-optimal control toolbox (OOP + TDD)."

@cli.command("di")
@click.option("--x0", type=float, required=True)
@click.option("--xf", type=float, required=True)
@click.option("--m", type=float, default=1.0)
@click.option("--F", "Fmax", type=float, default=10.0)
@click.option("--mu", type=float, default=0.0, help="Coulomb friction coefficient")
@click.option("--drag", type=float, default=0.0, help="Linear drag coefficient")
@click.option("--out", type=str, default="time/out")
def run_di(x0, xf, m, Fmax, mu, drag, out):
    """Minimum-time double-integrator (CasADi)."""
    res = DoubleIntegratorAPI().solve(x0=x0, xf=xf, m=m, F=Fmax, mu=mu, drag=drag, out_dir=out)
    click.echo(res.data)

@cli.command("2r-line")
@click.option("--y", type=float, required=True)
@click.option("--x0", type=float, required=True)
@click.option("--x1", type=float, required=True)
@click.option("--n", type=int, default=200)
@click.option("--tau", type=str, default="100,100", help="Tau limits as 'Pmax,Qmax'")
@click.option("--out", type=str, default="time/out")
def run_2r_line(y, x0, x1, n, tau, out):
    """Time-optimal along line y=const with TOPPRA (2R)."""
    tmax = tuple(map(float, tau.split(",")))
    res = TwoRAPI().line_y(y=y, x0=x0, x1=x1, N=n, tau_max=tmax, out_dir=out)
    click.echo(res.data)

@cli.command("diagram")
@click.option("--out", type=str, default="time/out/time.puml")
def diagram(out):
    """Emit a minimal PlantUML class diagram."""
    classes = ["TimeOptimalProblem", "MinTimeDoubleIntegrator", "TwoRPathTimeScaler", "Planar2RGeom"]
    edges = [("MinTimeDoubleIntegrator","TimeOptimalProblem"),
             ("TwoRPathTimeScaler","TimeOptimalProblem")]
    content = emit_puml(classes, edges)
    p = write_puml(out, content)
    click.echo(f"Wrote {p}")
