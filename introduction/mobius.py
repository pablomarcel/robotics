#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mobius.py — All-in-one Möbius strip CLI (mpl | plotly | pyvista)

- Backends:
    * mpl     → saves PNG (and can show a window with --show)
    * plotly  → saves HTML (PNG if kaleido is installed), opens in browser with --show
    * pyvista → saves PNG; uses off-screen render by default, or window with --show

- YAML config (e.g., introduction/in/mobius.yaml) can set defaults; CLI overrides YAML.

Examples (run from repo root)
-----------------------------
python introduction/mobius.py --backend mpl --R 1.0 --w 0.35
python introduction/mobius.py --backend plotly --R 1.0 --w 0.3
python introduction/mobius.py --backend pyvista --R 1.0 --w 0.35
python introduction/mobius.py --from-yaml introduction/in/mobius.yaml --w 0.25
python introduction/mobius.py --backend mpl --out introduction/out/my_mobius.png
"""

from __future__ import annotations
import argparse, os, sys
from typing import Optional, Dict, Any
import numpy as np

# Optional deps
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    MPL_OK = True
except Exception:
    MPL_OK = False

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    import pyvista as pv
    PYVISTA_OK = True
except Exception:
    PYVISTA_OK = False

try:
    import yaml
    YAML_OK = True
except Exception:
    YAML_OK = False


# --------------------
# Math: Möbius surface
# --------------------
def mobius_surface(R: float = 1.0, w: float = 0.35, nu: int = 400, nv: int = 60):
    """
    Standard 2-parameter Möbius strip:
      u ∈ [0, 2π), v ∈ [-w, w]
    x = (R + v*cos(u/2)) * cos(u)
    y = (R + v*cos(u/2)) * sin(u)
    z =  v*sin(u/2)
    """
    u = np.linspace(0.0, 2.0 * np.pi, int(nu))
    v = np.linspace(-w, w, int(nv))
    U, V = np.meshgrid(u, v)
    X = (R + V * np.cos(U / 2.0)) * np.cos(U)
    Y = (R + V * np.cos(U / 2.0)) * np.sin(U)
    Z = V * np.sin(U / 2.0)
    return X, Y, Z


# -------------
# I/O utilities
# -------------
def ensure_out_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def default_out_path(backend: str) -> str:
    os.makedirs("introduction/out", exist_ok=True)
    if backend == "plotly":
        return "introduction/out/mobius_plotly.html"
    ext = "png"
    return f"introduction/out/mobius_{backend}.{ext}"


# -------------
# Renderers
# -------------
def render_mpl(X, Y, Z, out_path: str, show: bool, opts: Dict[str, Any]):
    if not MPL_OK:
        sys.exit("Matplotlib is not available. Install with: pip install matplotlib")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    cmap = (opts or {}).get("cmap", "viridis")
    lw = float((opts or {}).get("linewidth", 0.15))
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="k", linewidth=lw, alpha=1.0)

    # roughly equal aspect
    max_range = (np.ptp(X) + np.ptp(Y) + np.ptp(Z)) / 3.0
    mid = np.array([X.mean(), Y.mean(), Z.mean()])
    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Möbius strip (Matplotlib)")
    ensure_out_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    print(f"[mpl] saved → {out_path}")
    if show:
        plt.show()


def render_plotly(X, Y, Z, out_path: str, show: bool, opts: Dict[str, Any]):
    if not PLOTLY_OK:
        sys.exit("Plotly is not available. Install with: pip install plotly")

    colorscale = (opts or {}).get("colorscale", "Turbo")
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale=colorscale)])
    fig.update_layout(title="Möbius strip (Plotly)", scene=dict(aspectmode="data"))

    ensure_out_dir(out_path)
    root, ext = os.path.splitext(out_path.lower())
    if ext in (".html", ".htm"):
        fig.write_html(out_path)
        print(f"[plotly] saved HTML → {out_path}")
    else:
        try:
            # PNG export requires kaleido
            fig.write_image(out_path, scale=2)
            print(f"[plotly] saved image → {out_path}")
        except Exception:
            fallback = root + ".html"
            fig.write_html(fallback)
            print(f"[plotly] could not write image, wrote HTML instead → {fallback}")

    if show:
        fig.show()


def render_pyvista(X, Y, Z, out_path: str, show: bool, opts: Dict[str, Any]):
    """
    IMPORTANT FIX:
    - We render & capture in a single call: plotter.show(screenshot=...).
      This guarantees a render has occurred (avoids 'Nothing to screenshot').
    - off_screen = not show  → no window unless --show is passed.
    """
    if not PYVISTA_OK:
        sys.exit("PyVista is not available. Install with: pip install pyvista")

    grid = pv.StructuredGrid(X, Y, Z)

    # Options with safe defaults
    cmap = (opts or {}).get("cmap", "viridis")
    smooth = bool((opts or {}).get("smooth_shading", True))
    specular = float((opts or {}).get("specular", 0.4))
    window = tuple((opts or {}).get("window", (1024, 768)))

    plotter = pv.Plotter(off_screen=not show, window_size=window)
    plotter.add_mesh(grid, cmap=cmap, smooth_shading=smooth, specular=specular)
    plotter.add_axes()
    plotter.show_bounds(grid="back", location="outer")

    ensure_out_dir(out_path)
    # Render AND save in one call; keep window open if --show was given.
    plotter.show(screenshot=out_path, auto_close=not show)
    print(f"[pyvista] saved screenshot → {out_path}")
    # If show=True, the above call opens a window and returns after it closes.


# -------------
# YAML handling
# -------------
def load_yaml(path: str) -> Dict[str, Any]:
    if not YAML_OK:
        sys.exit("PyYAML is not available. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        sys.exit("YAML must map keys to values (expected a dict).")
    return data


# -------------
# CLI parsing
# -------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mobius.py",
        description="Render a Möbius strip surface with mpl, plotly, or pyvista backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend", choices=["mpl", "plotly", "pyvista"], help="Rendering backend.")
    p.add_argument("--R", type=float, help="Ring radius.")
    p.add_argument("--w", type=float, help="Half-width of the strip.")
    p.add_argument("--nu", type=int, help="Samples along u (angle).")
    p.add_argument("--nv", type=int, help="Samples along v (width).")

    p.add_argument("--out", type=str, help="Output file path_planning (defaults into introduction/out/).")
    p.add_argument("--show", action="store_true", help="Show interactive window (mpl/pyvista).")

    p.add_argument("--from-yaml", type=str, help="Load defaults from YAML (e.g., introduction/in/mobius.yaml).")
    return p


def main(argv: Optional[list] = None):
    args = build_parser().parse_args(argv)

    # Defaults
    params: Dict[str, Any] = {
        "backend": "mpl",
        "R": 1.0,
        "w": 0.35,
        "nu": 400,
        "nv": 60,
        "mpl": {"cmap": "viridis", "linewidth": 0.15},
        "plotly": {"colorscale": "Turbo"},
        "pyvista": {"cmap": "viridis", "smooth_shading": True, "specular": 0.4, "window": (1024, 768)},
    }

    # YAML → merge
    if args.from_yaml:
        y = load_yaml(args.from_yaml)
        for k in ("backend", "R", "w", "nu", "nv"):
            if k in y:
                params[k] = y[k]
        for k in ("mpl", "plotly", "pyvista"):
            if k in y and isinstance(y[k], dict):
                params[k].update(y[k])

    # CLI overrides
    if args.backend: params["backend"] = args.backend
    if args.R is not None: params["R"] = float(args.R)
    if args.w is not None: params["w"] = float(args.w)
    if args.nu is not None: params["nu"] = int(args.nu)
    if args.nv is not None: params["nv"] = int(args.nv)

    backend = params["backend"]
    R, w, nu, nv = float(params["R"]), float(params["w"]), int(params["nu"]), int(params["nv"])

    # Output path_planning
    out_path = args.out if args.out else default_out_path(backend)
    if backend == "plotly" and os.path.splitext(out_path)[1] == "":
        out_path = out_path + ".html"
    if backend in ("mpl", "pyvista") and os.path.splitext(out_path)[1] == "":
        out_path = out_path + ".png"

    # Compute geometry
    X, Y, Z = mobius_surface(R=R, w=w, nu=nu, nv=nv)

    # Render
    if backend == "mpl":
        render_mpl(X, Y, Z, out_path, show=args.show, opts=params.get("mpl", {}))
    elif backend == "plotly":
        render_plotly(X, Y, Z, out_path, show=args.show, opts=params.get("plotly", {}))
    elif backend == "pyvista":
        render_pyvista(X, Y, Z, out_path, show=args.show, opts=params.get("pyvista", {}))
    else:
        sys.exit(f"Unknown backend: {backend}")

    # Summary
    print(f"Params: backend={backend}, R={R}, w={w}, nu={nu}, nv={nv}")
    print(f"Output: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
