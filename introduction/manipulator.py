#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manipulator.py — 72-configuration generator (R/P joints × axis relations)

Jazar (§1.3.1 Geometry): For 3-link, 2-adjacent-axis choices:
- Joints: each ∈ {R, P}  → 2^3 = 8
- Axis relation (J1↔J2, J2↔J3): each ∈ {parallel, orthogonal, perpendicular} → 3×3 = 9
Total combos = 8 × 9 = 72.

Features
========
- Filter by joint type at positions 1/2/3        (--j1/--j2/--j3)
- Filter by axis relation for 1↔2 and 2↔3        (--a12/--a23)
- Output ASCII, Unicode, or both                 (--style ascii|unicode|both)
- Pretty table with Rich (optional)              (--rich)
- Big ASCII banner with PyFiglet (optional)      (--figlet)
- Read one spec from YAML                        (--from-yaml introduction/in/manipulator.yaml)
- Write results to files                         (--out, --export-json, --export-csv)

Examples (from repo root)
=========================
  # 1) Show ALL 72 in a pretty table (Unicode), also save to text/JSON/CSV
  python introduction/manipulator.py --all --rich --style unicode \
      --out introduction/out/all_72.txt \
      --export-json introduction/out/all_72.json \
      --export-csv introduction/out/all_72.csv

  # 2) All configs that START with R
  python introduction/manipulator.py --j1 R --rich

  # 3) Second joint = P, any other
  python introduction/manipulator.py --j2 P

  # 4) Third joint = R, first axis parallel, second axis orthogonal
  python introduction/manipulator.py --j3 R --a12 parallel --a23 orthogonal --style both

  # 5) Read a manipulator spec from YAML and validate/print it
  #    (expects introduction/in/manipulator.yaml)
  python introduction/manipulator.py --from-yaml introduction/in/manipulator.yaml --style both --rich

YAML format
===========
joints: [R, R, P]           # or: type: "RRP"
axes:   [parallel, orthogonal]   # [a12, a23]

Notes
=====
- If Rich/PyFiglet are not installed, the script will fall back gracefully.
- Axis “perpendicular” here means skew-perpendicular (right angle w.r.t. common normal),
  distinct from “orthogonal” (intersecting at 90°) and “parallel” (∥), per Jazar’s text.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Tuple

# Optional prettifiers
try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except Exception:
    Console = None
    Table = None
    RICH_AVAILABLE = False

try:
    import pyfiglet
    PYFIGLET_AVAILABLE = True
except Exception:
    pyfiglet = None
    PYFIGLET_AVAILABLE = False

# Optional YAML
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    yaml = None
    YAML_AVAILABLE = False


# --------------------------
# Canonical domains & helpers
# --------------------------
JOINTS = ("R", "P")

AXIS_CANON = ("parallel", "orthogonal", "perpendicular")

AXIS_SYNONYMS = {
    "||": "parallel",
    "par": "parallel",
    "parallel": "parallel",

    "orth": "orthogonal",
    "ortho": "orthogonal",
    "orthogonal": "orthogonal",
    "⊥": "orthogonal",

    "perp": "perpendicular",
    "perpendicular": "perpendicular",
    "⟂": "perpendicular",
}

JOINT_SYNONYMS = {
    "r": "R", "rev": "R", "revolute": "R", "R": "R",
    "p": "P", "pri": "P", "prismatic": "P", "P": "P",
}

AXIS_GLYPHS = {
    # Unicode glyphs (pretty)
    "unicode": {
        "parallel": "∥",
        "orthogonal": "⊥",
        "perpendicular": "⟂",
    },
    # ASCII-friendly (no fancy glyphs)
    "ascii": {
        "parallel": "||",
        "orthogonal": "ORTH",
        "perpendicular": "PERP",
    },
    # Very short codes (for filenames, etc.)
    "short": {
        "parallel": "PAR",
        "orthogonal": "ORTH",
        "perpendicular": "PERP",
    },
}


def norm_joint(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    key = str(s).strip()
    return JOINT_SYNONYMS.get(key, key if key in JOINTS else None)


def norm_axis(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    key = str(s).strip().lower()
    return AXIS_SYNONYMS.get(key, key if key in AXIS_CANON else None)


@dataclass(frozen=True)
class Config:
    j1: str
    j2: str
    j3: str
    a12: str
    a23: str

    # symbolic renderings
    def render(self, style: str = "unicode") -> str:
        """
        style ∈ {"ascii", "unicode", "both"}
        """
        if style not in {"ascii", "unicode", "both"}:
            style = "unicode"

        uni = f"{self.j1} {AXIS_GLYPHS['unicode'][self.a12]} {self.j2} {AXIS_GLYPHS['unicode'][self.a23]} {self.j3}"
        asc = f"{self.j1} {AXIS_GLYPHS['ascii'][self.a12]} {self.j2} {AXIS_GLYPHS['ascii'][self.a23]} {self.j3}"

        if style == "ascii":
            return asc
        elif style == "both":
            return f"{uni}  |  {asc}"
        else:
            return uni

    def as_dict(self) -> dict:
        d = asdict(self)
        d["unicode"] = self.render("unicode")
        d["ascii"] = self.render("ascii")
        d["label"] = f"{self.j1}{AXIS_GLYPHS['short'][self.a12]}{self.j2}{AXIS_GLYPHS['short'][self.a23]}{self.j3}"
        return d


def all_72() -> List[Config]:
    return [
        Config(j1, j2, j3, a12, a23)
        for (j1, j2, j3) in itertools.product(JOINTS, repeat=3)
        for (a12, a23) in itertools.product(AXIS_CANON, repeat=2)
    ]


def matches(cfg: Config,
            j1: Optional[str], j2: Optional[str], j3: Optional[str],
            a12: Optional[str], a23: Optional[str]) -> bool:
    if j1 and cfg.j1 != j1:
        return False
    if j2 and cfg.j2 != j2:
        return False
    if j3 and cfg.j3 != j3:
        return False
    if a12 and cfg.a12 != a12:
        return False
    if a23 and cfg.a23 != a23:
        return False
    return True


def filter_configs(universe: Iterable[Config],
                   j1: Optional[str], j2: Optional[str], j3: Optional[str],
                   a12: Optional[str], a23: Optional[str]) -> List[Config]:
    return [c for c in universe if matches(c, j1, j2, j3, a12, a23)]


def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# --------------------------
# I/O: Text / JSON / CSV / YAML
# --------------------------
def write_text(path: str, configs: List[Config], style: str, banner: Optional[str] = None):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        if banner:
            f.write(banner + "\n")
        for i, c in enumerate(configs, 1):
            f.write(f"{i:02d}. {c.render(style)}   [joints={c.j1}{c.j2}{c.j3}, a12={c.a12}, a23={c.a23}]\n")


def write_json(path: str, configs: List[Config]):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.as_dict() for c in configs], f, indent=2, ensure_ascii=False)


def write_csv(path: str, configs: List[Config]):
    ensure_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["j1", "j2", "j3", "a12", "a23", "unicode", "ascii", "label"])
        w.writeheader()
        for c in configs:
            w.writerow(c.as_dict())


def read_yaml(path: str) -> Config:
    if not YAML_AVAILABLE:
        raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Accept either "joints: [R,R,P]" or "type: RRP"
    if "joints" in data:
        j1, j2, j3 = data["joints"]
    elif "type" in data:
        t = str(data["type"]).strip()
        if len(t) != 3:
            raise ValueError("YAML 'type' must be a 3-character string like 'RRP'.")
        j1, j2, j3 = t[0], t[1], t[2]
    else:
        raise ValueError("YAML must provide 'joints' or 'type'.")

    if "axes" not in data or len(data["axes"]) != 2:
        raise ValueError("YAML must provide 'axes' as [a12, a23].")

    a12, a23 = data["axes"]

    # Normalize & validate
    j1n, j2n, j3n = norm_joint(j1), norm_joint(j2), norm_joint(j3)
    a12n, a23n = norm_axis(a12), norm_axis(a23)
    if not (j1n and j2n and j3n and a12n and a23n):
        raise ValueError("Invalid joint or axis values in YAML.")
    return Config(j1n, j2n, j3n, a12n, a23n)


# --------------------------
# Pretty printing
# --------------------------
def make_banner(text: str) -> Optional[str]:
    if not PYFIGLET_AVAILABLE:
        return None
    try:
        return pyfiglet.figlet_format(text)
    except Exception:
        return text


def print_table(configs: List[Config], style: str, title: str = "Manipulator Geometries"):
    use_rich = RICH_AVAILABLE
    if not use_rich:
        # Plain fallback
        print(f"\n{title} ({len(configs)} results)\n" + "-" * 60)
        for i, c in enumerate(configs, 1):
            print(f"{i:02d}. {c.render(style)}   [joints={c.j1}{c.j2}{c.j3}, a12={c.a12}, a23={c.a23}]")
        return

    console = Console()
    table = Table(title=f"{title} ({len(configs)} results)")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Symbol", style="bold")
    table.add_column("Joints", justify="center")
    table.add_column("a12", justify="center")
    table.add_column("a23", justify="center")
    table.add_column("Label", justify="center", style="dim")

    for i, c in enumerate(configs, 1):
        table.add_row(
            f"{i:02d}",
            c.render(style),
            f"{c.j1}{c.j2}{c.j3}",
            c.a12,
            c.a23,
            c.as_dict()["label"],
        )
    console.print(table)


# --------------------------
# CLI
# --------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manipulator.py",
        description="Generate and filter the 72 industrial manipulator geometries (R/P joints × axis relations).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Filters
    p.add_argument("--j1", type=str, help="First joint type (R/P).")
    p.add_argument("--j2", type=str, help="Second joint type (R/P).")
    p.add_argument("--j3", type=str, help="Third joint type (R/P).")

    p.add_argument("--a12", type=str, help="Axis relation between J1 and J2 "
                                           "(parallel/orthogonal/perpendicular).")
    p.add_argument("--a23", type=str, help="Axis relation between J2 and J3 "
                                           "(parallel/orthogonal/perpendicular).")

    p.add_argument("--style", choices=["ascii", "unicode", "both"], default="unicode",
                   help="Rendering style for the symbolic column.")
    p.add_argument("--rich", action="store_true", help="Pretty table (optional).")
    p.add_argument("--figlet", action="store_true", help="Show an ASCII banner (optional).")
    p.add_argument("--count", action="store_true", help="Only print the count of results.")

    # Modes
    p.add_argument("--all", action="store_true", help="Ignore filters and show all 72.")
    p.add_argument("--from-yaml", type=str, help="Read one manipulator spec from YAML.")

    # Output
    p.add_argument("--out", type=str, help="Write a text listing to this path_planning.")
    p.add_argument("--export-json", type=str, help="Also write a JSON file with results.")
    p.add_argument("--export-csv", type=str, help="Also write a CSV file with results.")

    return p


def main(argv: Optional[List[str]] = None):
    args = build_argparser().parse_args(argv)

    # Banner
    banner = None
    if args.figlet:
        banner = make_banner("Manipulator 72")
        if banner:
            print(banner)

    # Normalize filters
    j1 = norm_joint(args.j1)
    j2 = norm_joint(args.j2)
    j3 = norm_joint(args.j3)
    a12 = norm_axis(args.a12)
    a23 = norm_axis(args.a23)

    # Source set
    universe = all_72()

    # YAML mode (validate & print that one)
    if args.from_yaml:
        cfg = read_yaml(args.from_yaml)
        # Confirm it's part of the 72 (it always will be if domains are valid)
        if args.count:
            print(1)
            return
        if args.rich:
            print_table([cfg], style=args.style, title="YAML Manipulator")
        else:
            print(f"YAML Manipulator: {cfg.render(args.style)}  "
                  f"[joints={cfg.j1}{cfg.j2}{cfg.j3}, a12={cfg.a12}, a23={cfg.a23}]")
        # Optional outputs
        if args.out:
            write_text(args.out, [cfg], style=args.style, banner=banner)
        if args.export_json:
            write_json(args.export_json, [cfg])
        if args.export_csv:
            write_csv(args.export_csv, [cfg])
        return

    # Regular listing (all or filtered)
    if args.all:
        results = universe
    else:
        results = filter_configs(universe, j1, j2, j3, a12, a23)

    # Count-only mode
    if args.count:
        print(len(results))
        return

    # Print table or plain
    title = "Manipulator Geometries"
    if args.rich:
        print_table(results, style=args.style, title=title)
    else:
        print(f"{title} ({len(results)} results)")
        print("-" * 60)
        for i, c in enumerate(results, 1):
            print(f"{i:02d}. {c.render(args.style)}   [joints={c.j1}{c.j2}{c.j3}, a12={c.a12}, a23={c.a23}]")

    # Optional file outputs
    if args.out:
        write_text(args.out, results, style=args.style, banner=banner)
    if args.export_json:
        write_json(args.export_json, results)
    if args.export_csv:
        write_csv(args.export_csv, results)


if __name__ == "__main__":
    main()
