#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mobius_pyvista.py — Render a Möbius strip with PyVista and save a screenshot.

Fix for "Nothing to screenshot": we render with plotter.show(screenshot=...)
and (optionally) run off-screen to avoid opening a window.

Examples (from repo root):
  # Headless render (no window), PNG to introduction/out/
  python introduction/mobius_pyvista.py

  # Show an interactive window AND save a screenshot
  python introduction/mobius_pyvista.py --show

  # Custom geometry, resolution, window size, output
  python introduction/mobius_pyvista.py --R 1.0 --w 0.35 --nu 500 --nv 120 \
      --window 1280 900 --out introduction/out/mobius_custom.png
"""
import argparse
import os
import numpy as np
import pyvista as pv


def mobius_uv(R: float = 1.0, w: float = 0.35, nu: int = 400, nv: int = 80):
    """
    2-parameter Möbius strip surface:
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
    Z =  V * np.sin(U / 2.0)
    return X, Y, Z


def ensure_out_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def build_grid(R=1.0, w=0.35, nu=400, nv=80) -> pv.StructuredGrid:
    X, Y, Z = mobius_uv(R=R, w=w, nu=nu, nv=nv)
    # StructuredGrid accepts 2D arrays (nv×nu) for x,y,z
    return pv.StructuredGrid(X, Y, Z)


def render_and_save(grid: pv.StructuredGrid, out_path: str, show_window: bool,
                    window_size=(1024, 768), cmap="viridis",
                    smooth_shading=True, specular=0.4):
    """
    Render with PyVista and save a screenshot.
    - If show_window is False → off-screen render, no GUI.
    - If show_window is True  → on-screen window stays open after saving.
    """
    ensure_out_dir(out_path)

    plotter = pv.Plotter(off_screen=not show_window, window_size=window_size)
    plotter.add_mesh(grid, cmap=cmap, smooth_shading=smooth_shading, specular=specular)
    plotter.add_axes()
    plotter.show_bounds(grid="back", location="outer")

    # IMPORTANT: render while saving
    # auto_close=False keeps the window open if show_window=True
    plotter.show(screenshot=out_path, auto_close=not show_window)

    print(f"[pyvista] saved screenshot → {out_path}")

    # If the window is still open (show_window=True), the call above already displayed it.
    # If you want to close programmatically, you could call plotter.close() here.


def parse_args():
    p = argparse.ArgumentParser(
        description="Render a Möbius strip with PyVista and save a screenshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--R", type=float, default=1.0, help="Ring radius")
    p.add_argument("--w", type=float, default=0.35, help="Half-width of strip")
    p.add_argument("--nu", type=int, default=400, help="Samples along u (angle)")
    p.add_argument("--nv", type=int, default=80, help="Samples along v (width)")
    p.add_argument("--out", type=str, default="introduction/out/mobius_pyvista.png",
                   help="Output image path_planning (.png recommended)")
    p.add_argument("--window", type=int, nargs=2, metavar=("W", "H"),
                   default=(1024, 768), help="Render window size (pixels)")
    p.add_argument("--cmap", type=str, default="viridis", help="Colormap")
    p.add_argument("--no-smooth", action="store_true", help="Disable smooth shading")
    p.add_argument("--specular", type=float, default=0.4, help="Specular intensity")
    p.add_argument("--show", action="store_true",
                   help="Show interactive window (on-screen). If omitted, renders off-screen.")
    return p.parse_args()


def main():
    args = parse_args()
    grid = build_grid(R=args.R, w=args.w, nu=args.nu, nv=args.nv)
    render_and_save(
        grid=grid,
        out_path=args.out,
        show_window=args.show,
        window_size=tuple(args.window),
        cmap=args.cmap,
        smooth_shading=not args.no_smooth,
        specular=args.specular,
    )
    print(f"Params: R={args.R}, w={args.w}, nu={args.nu}, nv={args.nv}, show={args.show}")


if __name__ == "__main__":
    main()
