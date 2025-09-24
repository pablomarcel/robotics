# control/cli.py
from __future__ import annotations
import json
import click
import numpy as np

from .app import ControlApp
from .io import JsonStore
from .tools.diagram_tool import emit_mermaid


@click.group(help="Control Techniques CLI (14.1–14.118)")
def app():
    pass


@app.command("msd_pd")
@click.option("--m",  type=float, default=1.0, show_default=True)
@click.option("--c",  type=float, default=0.8, show_default=True)
@click.option("--k",  type=float, default=10.0, show_default=True)
@click.option("--kp", type=float, default=30.0, show_default=True)
@click.option("--kd", type=float, default=10.0, show_default=True)
# Support BOTH --t and --T (test uses --T)
@click.option("--t", "--T", "T", type=float, default=3.0, show_default=True)
@click.option("--x0", type=str, default="1,0", show_default=True,
              help="Initial state 'x,xdot'")
@click.option("--out", type=str, default="msd_pd", show_default=True)
def msd_pd_cmd(m, c, k, kp, kd, T, x0, out):
    """Closed-loop mass–spring–damper with PD (maps to 14.10/14.12–14.25)."""
    ca = ControlApp()
    plant = ca.msd(m, c, k)
    x0v = np.array([float(s) for s in x0.split(",")], dtype=float)

    # Simple constant PD on the initial error (smoke test friendly)
    def u_fun(_t):
        e, ed = x0v[0], x0v[1]
        return -kd * ed - kp * e

    res = ca.api.simulate_msd(plant, u_fun, (0.0, T), x0v)
    JsonStore().write(out, {"t": res.t.tolist(), "x": res.x.tolist()})
    click.echo(f"Wrote out/{out}.json")


@app.command("pendulum_pid")
@click.option("--t", "T", type=float, default=4.0, show_default=True)
@click.option("--kp", type=float, default=30.0, show_default=True)
@click.option("--ki", type=float, default=5.0,  show_default=True)
@click.option("--kd", type=float, default=10.0, show_default=True)
@click.option("--out", type=str, default="pend_pid", show_default=True)
def pendulum_pid_cmd(T, kp, ki, kd, out):
    """PID about θd=π/2 (14.80–14.88). Uses ControlApp canned scenario."""
    ca = ControlApp()
    res = ca.run_pendulum_pid_at_pi_over_2(T=T)
    JsonStore().write(out, {"t": res.t.tolist(), "x": res.x.tolist()})
    click.echo(f"Wrote out/{out}.json")


@app.command("robot_ct")
@click.option("--q",      type=str, default="0,0", show_default=True)
@click.option("--qd",     type=str, default="0,0", show_default=True)
@click.option("--qd-d",   "qd_d",   type=str, default="0,0", show_default=True)
@click.option("--qdd-d",  "qdd_d",  type=str, default="0,0", show_default=True)
@click.option("--wn",     type=float, default=4.0, show_default=True)
@click.option("--zeta",   type=float, default=1.0, show_default=True)
def robot_ct_cmd(q, qd, qd_d, qdd_d, wn, zeta):
    """Computed-torque PD for planar 2R (14.33/14.41/14.93–14.95)."""
    to_vec = lambda s: np.array([float(v) for v in s.split(",")], dtype=float)
    q, qd, qd_d, qdd_d = map(to_vec, (q, qd, qd_d, qdd_d))
    ca = ControlApp()
    out = ca.api.robot_computed_torque(ca.planar2r(), q, qd, q, qd_d, qdd_d, wn=wn, zeta=zeta)
    click.echo(json.dumps(
        {"tau": out["tau"].tolist(), "kp": out["kp"].tolist(), "kd": out["kd"].tolist()},
        indent=2
    ))


@app.command("diagram")
@click.option("--out", "out_name", type=str, default="classes", show_default=True)
def diagram_cmd(out_name):
    """Emit a Mermaid class diagram Markdown to control/out/{out}.md."""
    p = emit_mermaid(out_name=out_name)
    click.echo(f"Wrote {p}")


if __name__ == "__main__":
    app()
