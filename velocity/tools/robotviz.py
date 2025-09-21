# velocity/tools/robotviz.py
"""
RobotViz — minimal DH robot stick-figure renderer (SVG/PNG/PDF).

New in this version
-------------------
- --fmt svg|png|pdf (default: svg)
- PNG/PDF rendering via optional 'cairosvg' (pure Python)
- --dpi for raster density, or --width/--height to scale to exact pixels/points
- --transparent to drop the solid background rect (useful for docs/themes)

What it does
------------
- Loads a DH robot spec (YAML or JSON) compatible with velocity.core.DHRobot.from_spec
- Runs FK at a provided configuration q (defaults to zeros)
- Projects joints to a view plane (xy/xz/yz/isometric)
- Renders a clean diagram: links as segments, joints as circles, TCP & base as squares

CLI Examples (run from repo root)
---------------------------------
# 1) SVG (no extra deps)
python -m velocity.tools.robotviz draw velocity/in/planar2r.yaml --out velocity/out/planar2r.svg

# 2) PNG (requires 'pip install cairosvg')
python -m velocity.tools.robotviz draw velocity/in/planar2r.yaml --fmt png --out velocity/out/planar2r.png

# 3) PNG with configuration, dark theme, higher DPI
python -m velocity.tools.robotviz draw velocity/in/planar2r.yaml \
  --q 0.3,-0.4 --theme dark --fmt png --dpi 300 \
  --out velocity/out/planar2r_dark.png

# 4) Exact size PNG (width/height in pixels); DPI ignored when width/height provided
python -m velocity.tools.robotviz draw velocity/in/planar2r.yaml \
  --fmt png --width 1200 --height 900 --out velocity/out/planar2r_1200x900.png

# 5) Transparent background (SVG or PNG)
python -m velocity.tools.robotviz draw velocity/in/planar2r.yaml \
  --transparent --fmt png --out velocity/out/planar2r_transparent.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Literal

import numpy as np

from velocity import core

# ------------------------------ Themes / Styles --------------------------------

_THEMES = {
    "light": dict(
        bg="#ffffff",
        link="#111827",
        joint_fill="#f59e0b",
        joint_stroke="#111827",
        tcp_fill="#10b981",
        tcp_stroke="#065f46",
        base_fill="#3b82f6",
        base_stroke="#1e40af",
        axis="#9ca3af",
    ),
    "dark": dict(
        bg="#0f172a",
        link="#cbd5e1",
        joint_fill="#f59e0b",
        joint_stroke="#cbd5e1",
        tcp_fill="#10b981",
        tcp_stroke="#34d399",
        base_fill="#60a5fa",
        base_stroke="#93c5fd",
        axis="#64748b",
    ),
}


# ------------------------------ Data / Config ----------------------------------

@dataclass(frozen=True)
class VizConfig:
    view: str = "xy"          # 'xy' | 'xz' | 'yz' | 'iso'
    px_per_m: float = 220.0   # scale (pixels per meter)
    pad_px: int = 30          # padding around drawing in pixels
    joint_radius_px: int = 5
    tcp_size_px: int = 8
    stroke_px: float = 2.0
    theme: str = "light"
    show_axes: bool = True
    axes_len_m: float = 0.2   # 20 cm axes from base
    transparent: bool = False # omit solid background rect when True


# ------------------------------ IO helpers -------------------------------------

def _load_spec(path: Path) -> dict:
    """
    Load a robot spec (YAML or JSON). The format must match core.DHRobot.from_spec.
    """
    s = path.suffix.lower()
    if s in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise SystemExit("PyYAML is required to load YAML files. Try: pip install pyyaml") from e
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    elif s == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"Unsupported spec extension '{path.suffix}'. Use .yaml/.yml or .json")


def _parse_q(arg: Optional[str], dof: int) -> np.ndarray:
    if arg is None or arg.strip() == "":
        return np.zeros(dof, dtype=float)
    vals = [float(x) for x in arg.split(",")]
    if len(vals) != dof:
        raise SystemExit(f"--q has {len(vals)} values, but robot has {dof} joints")
    return np.asarray(vals, dtype=float)


# ------------------------------ Geometry / View --------------------------------

def _project_points(P: np.ndarray, view: str) -> np.ndarray:
    """
    Project 3D points (N,3) to 2D (N,2) using a simple view:
    - 'xy': drop z
    - 'xz': (x,z)
    - 'yz': (y,z)
    - 'iso': oblique isometric-ish projection
    """
    view = view.lower()
    if view == "xy":
        return P[:, [0, 1]]
    if view == "xz":
        return P[:, [0, 2]]
    if view == "yz":
        return P[:, [1, 2]]

    if view in ("iso", "isometric"):
        # Oblique projection matrix to 2D
        # (x, y, z) -> (x - z/√2, y - z/√2)
        a = 1.0 / np.sqrt(2.0)
        M = np.array([[1.0, 0.0, -a],
                      [0.0, 1.0, -a]], dtype=float)
        return (M @ P.T).T

    raise SystemExit(f"Unknown view '{view}'. Use xy|xz|yz|iso")


def _world_to_canvas(P2: np.ndarray, cfg: VizConfig) -> Tuple[np.ndarray, int, int]:
    """
    Map 2D world coords to SVG pixel coords with padding and Y-up -> Y-down flip.
    Returns (points_px, width, height).
    """
    if P2.size == 0:
        W = H = 2 * cfg.pad_px + 10
        return np.zeros((0, 2)), W, H

    # Scale
    S = cfg.px_per_m
    Q = P2 * S

    # Compute bounding box
    min_xy = np.min(Q, axis=0)
    max_xy = np.max(Q, axis=0)
    size = (max_xy - min_xy)
    W = int(size[0] + 2 * cfg.pad_px)
    H = int(size[1] + 2 * cfg.pad_px)
    if W < 2 * cfg.pad_px + 10: W = 2 * cfg.pad_px + 10
    if H < 2 * cfg.pad_px + 10: H = 2 * cfg.pad_px + 10

    # Translate so min -> pad, and flip Y (SVG is Y-down)
    Q = Q - min_xy + np.array([cfg.pad_px, cfg.pad_px], dtype=float)
    Q[:, 1] = H - Q[:, 1]
    return Q, W, H


# ------------------------------ SVG helpers ------------------------------------

def _svg_header(w: int, h: int, bg: str, transparent: bool) -> List[str]:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    if not transparent:
        out.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{bg}"/>')
    return out


def _svg_footer() -> List[str]:
    return ["</svg>"]


def _svg_line(x1, y1, x2, y2, color, width=2.0) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" ' \
           f'stroke="{color}" stroke-width="{width:.2f}" stroke-linecap="round" />'


def _svg_circle(cx, cy, r, fill, stroke, sw=1.5) -> str:
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}" />'


def _svg_square(cx, cy, s, fill, stroke, sw=1.5) -> str:
    x = cx - s / 2
    y = cy - s / 2
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{s:.2f}" height="{s:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}" />'


def _svg_text(x, y, txt, color, size=11, anchor="middle") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-size="{size}" text-anchor="{anchor}" ' \
           f'font-family="Helvetica,Arial,sans-serif">{txt}</text>'


# ------------------------------ Core rendering ---------------------------------

def render_robot_svg(spec: dict, q: Sequence[float], cfg: VizConfig) -> Tuple[str, int, int]:
    """
    Build SVG string for the given robot spec and configuration.
    Returns: (svg_text, width_px, height_px)
    """
    robot = core.DHRobot.from_spec(spec)
    n = len(robot.joints)
    qv = np.asarray(q, dtype=float).reshape(n)

    frames, _ = robot._fk_all(qv)          # frames includes base and final TCP
    pts = np.stack([F[:3, 3] for F in frames], axis=0)  # (n+1, 3)

    # Choose view and project
    P2 = _project_points(pts, cfg.view)
    Ppx, W, H = _world_to_canvas(P2, cfg)

    colors = _THEMES.get(cfg.theme, _THEMES["light"])
    out: List[str] = []
    out += _svg_header(W, H, colors["bg"], cfg.transparent)

    # Optional base axes
    if cfg.show_axes:
        base = Ppx[0]
        if cfg.view in ("xy", "xz", "yz"):
            axes = np.array([[cfg.axes_len_m, 0.0],
                             [0.0, cfg.axes_len_m]])
            axes_px = axes * cfg.px_per_m
            x_end = base + np.array([axes_px[0, 0], -axes_px[0, 1]])
            y_end = base + np.array([axes_px[1, 0], -axes_px[1, 1]])
        else:  # iso
            a = 1.0 / np.sqrt(2.0)
            axes3 = np.array([[cfg.axes_len_m, 0, 0],
                              [0, cfg.axes_len_m, 0],
                              [0, 0, cfg.axes_len_m]])
            axes2 = (np.array([[1.0, 0.0, -a],
                               [0.0, 1.0, -a]]) @ axes3.T).T
            axes_px = axes2 * cfg.px_per_m
            x_end = base + np.array([axes_px[0, 0], -axes_px[0, 1]])
            y_end = base + np.array([axes_px[1, 0], -axes_px[1, 1]])

        out.append(_svg_line(base[0], base[1], x_end[0], x_end[1], colors["axis"], width=1.5))
        out.append(_svg_line(base[0], base[1], y_end[0], y_end[1], colors["axis"], width=1.5))
        out.append(_svg_text(x_end[0] + 8, x_end[1] - 4, "x", colors["axis"], size=10))
        out.append(_svg_text(y_end[0] + 8, y_end[1] - 4, "y", colors["axis"], size=10))

    # Links
    for i in range(len(Ppx) - 1):
        x1, y1 = Ppx[i]
        x2, y2 = Ppx[i + 1]
        out.append(_svg_line(x1, y1, x2, y2, colors["link"], width=cfg.stroke_px))

    # Joints
    for i in range(len(Ppx) - 1):
        cx, cy = Ppx[i]
        out.append(_svg_circle(cx, cy, cfg.joint_radius_px, colors["joint_fill"], colors["joint_stroke"], sw=1.5))
        out.append(_svg_text(cx, cy - (cfg.joint_radius_px + 6), f"J{i+1}", colors["joint_stroke"], size=10))

    # Base marker (square) at frames[0]
    bx, by = Ppx[0]
    out.append(_svg_square(bx, by, cfg.tcp_size_px, colors["base_fill"], colors["base_stroke"], sw=1.4))
    out.append(_svg_text(bx, by + (cfg.tcp_size_px + 10), "base", colors["base_stroke"], size=10))

    # TCP marker (square) at last point
    ex, ey = Ppx[-1]
    out.append(_svg_square(ex, ey, cfg.tcp_size_px, colors["tcp_fill"], colors["tcp_stroke"], sw=1.6))
    out.append(_svg_text(ex, ey - (cfg.tcp_size_px + 8), "TCP", colors["tcp_stroke"], size=10))

    out += _svg_footer()
    svg = "\n".join(out)
    return svg, W, H


# ------------------------------ Output helpers ---------------------------------

def _write_svg(svg: str, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return str(out_path)


def _write_png_pdf(svg: str, out_path: Path, fmt: Literal["png", "pdf"], *,
                   dpi: int = 200, width: Optional[int] = None, height: Optional[int] = None) -> str:
    """
    Render PNG/PDF using cairosvg if available. Width/height take precedence over DPI.
    PNG: width/height are pixels; PDF: width/height are points.
    """
    try:
        import cairosvg  # type: ignore
    except Exception as e:
        raise SystemExit(
            "PNG/PDF output requires 'cairosvg'. Install it with:\n"
            "  pip install cairosvg"
        ) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(out_path),
            dpi=dpi if (width is None and height is None) else None,
            output_width=width,
            output_height=height,
            background_color=None,  # keep transparency if no <rect>
        )
    else:  # pdf
        cairosvg.svg2pdf(
            bytestring=svg.encode("utf-8"),
            write_to=str(out_path),
            dpi=dpi if (width is None and height is None) else None,
            output_width=width,
            output_height=height,
            background_color=None,
        )
    return str(out_path)


# ------------------------------ CLI plumbing -----------------------------------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="velocity.tools.robotviz", description="Render a DH robot spec as SVG/PNG/PDF")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draw", help="Render diagram")
    d.add_argument("robot", help="Path to YAML/JSON DH robot spec (velocity/in/...)")
    d.add_argument("--q", default="", help="Comma-separated joint values (rad/m). Default: zeros")

    d.add_argument("--fmt", default="svg", choices=["svg", "png", "pdf"], help="Output format")
    d.add_argument("--out", default="", help="Output file path; if blank, uses velocity/out/<name>.<fmt>")

    d.add_argument("--view", default="xy", choices=["xy", "xz", "yz", "iso"], help="Projection plane")
    d.add_argument("--scale", type=float, default=220.0, help="Pixels per meter (SVG logical units)")
    d.add_argument("--stroke", type=float, default=2.0, help="Link stroke width (px)")
    d.add_argument("--theme", default="light", choices=list(_THEMES.keys()))
    d.add_argument("--transparent", action="store_true", help="Omit solid background rect")

    # Raster sizing (PNG/PDF)
    d.add_argument("--dpi", type=int, default=220, help="Raster DPI when width/height are not specified")
    d.add_argument("--width", type=int, default=None, help="Target width (pixels for PNG, points for PDF)")
    d.add_argument("--height", type=int, default=None, help="Target height (pixels for PNG, points for PDF)")

    d.add_argument("--no-axes", action="store_true", help="Hide base axes")
    d.add_argument("--axes-len", type=float, default=0.2, help="Base axes length (meters)")
    return p


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.cmd == "draw":
        spec_path = Path(args.robot)
        spec = _load_spec(spec_path)

        # We need DOF to parse q
        tmp_robot = core.DHRobot.from_spec(spec)
        q = _parse_q(args.q, len(tmp_robot.joints))

        cfg = VizConfig(
            view=args.view,
            px_per_m=float(args.scale),
            stroke_px=float(args.stroke),
            theme=args.theme,
            show_axes=not args.no_axes,
            axes_len_m=float(args.axes_len),
            transparent=bool(args.transparent),
        )

        svg, W, H = render_robot_svg(spec, q, cfg)

        # Choose default output if not provided
        if not args.out:
            name = spec.get("name", "robot")
            out_dir = Path("velocity/out")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{name}.{args.fmt}"
        else:
            out_path = Path(args.out)

        fmt = args.fmt.lower()
        if fmt == "svg":
            print(_write_svg(svg, out_path))
        elif fmt in ("png", "pdf"):
            print(_write_png_pdf(svg, out_path, fmt=fmt, dpi=int(args.dpi),
                                 width=args.width, height=args.height))
        else:
            raise SystemExit(f"Unknown format: {fmt}")
        return


if __name__ == "__main__":  # pragma: no cover
    main()
