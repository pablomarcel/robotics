#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kinematics.py — Jazar §1.5 vector & frame toolkit (CLI)

Ops (use with --op):
  unit, magnitude, natural, decompose, dircos, dot, cross, add, laws,
  triple_scalar, triple_vector, nd_scalar_det, ortho_check, gram_schmidt,
  transform, rotmat, euler_from_R, norm

Robust zsh-friendly parsing:
  You can pass vectors as:  --v1 "1,2,3"  or  --v1 1 2 3  or  --v1=1,2,3
  Same for --v2/--v3; --euler takes 3 numbers; --Rmat takes 9 numbers.

YAML batch:
  --from-yaml intro/in/calcs.yaml   (contains a list of tasks)

Optional file logging:
  --out-txt intro/out/results.txt   --out-csv intro/out/results.csv
"""

from __future__ import annotations
import argparse, csv, os, sys, textwrap
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import sympy as sp

# Pretty console
try:
    from rich.console import Console
    from rich.table import Table
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None

# YAML
try:
    import yaml
    YAML_OK = True
except Exception:
    YAML_OK = False


# ----------------------------
# Utilities & robust parsing
# ----------------------------
def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _tokens_to_floats(tokens: List[str]) -> List[float]:
    vals: List[float] = []
    for t in tokens:
        if "," in t:
            for p in t.split(","):
                p = p.strip()
                if p:
                    vals.append(float(p))
        else:
            vals.append(float(t))
    return vals

def parse_vec_any(arg: Optional[Any]) -> Optional[np.ndarray]:
    """
    Accept:
      - None
      - string "1,2,3" or "1 2 3"
      - list like ["1","2","3"]  (from nargs=+)
      - equals syntax handled by argparse as a single string
    """
    if arg is None:
        return None
    if isinstance(arg, np.ndarray):
        return arg.astype(float).ravel()
    if isinstance(arg, (list, tuple)):
        vals = _tokens_to_floats([str(x) for x in arg])
        return np.array(vals, dtype=float).ravel()
    # string
    s = str(arg).replace(",", " ")
    parts = [p for p in s.split() if p]
    vals = [float(p) for p in parts]
    return np.array(vals, dtype=float).ravel()

def parse_mat3_any(arg: Optional[Any]) -> Optional[np.ndarray]:
    """
    Accept 9 numbers: as string "a,..,i" or 9 separate tokens.
    """
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        vals = _tokens_to_floats([str(x) for x in arg])
    else:
        vals = _tokens_to_floats([str(arg)])
    if len(vals) != 9:
        raise ValueError("--Rmat requires 9 numbers (row-major).")
    return np.array(vals, dtype=float).reshape(3, 3)

def parse_angles_any(arg: Optional[Any]) -> Optional[np.ndarray]:
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        vals = _tokens_to_floats([str(x) for x in arg])
    else:
        vals = _tokens_to_floats([str(arg)])
    if len(vals) != 3:
        raise ValueError("--euler requires 3 numbers.")
    return np.array(vals, dtype=float)

def rich_table(title: str, rows: List[Tuple[str, str]]):
    if not RICH:
        print(f"\n{title}\n" + "-"*len(title))
        for k, v in rows:
            print(f"{k:>18s} : {v}")
        return
    t = Table(title=title, show_lines=True)
    t.add_column("Quantity", style="cyan", no_wrap=True)
    t.add_column("Value", style="white")
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)

def to_list_str(x: np.ndarray, fmt: str="{:.6g}") -> str:
    return "[" + ", ".join(fmt.format(v) for v in x.ravel()) + "]"


# ----------------------------
# Core vector computations
# ----------------------------
def magnitude(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def unit(v: np.ndarray, eps=1e-12) -> np.ndarray:
    n = magnitude(v)
    if n < eps:
        raise ValueError("Zero vector has no direction for unit().")
    return v / n

def natural_form(v: np.ndarray) -> Tuple[float, np.ndarray]:
    n = magnitude(v)
    return n, (v / n if n > 0 else v)

def decomposition(v: np.ndarray) -> Tuple[float, float, float]:
    if v.size != 3:
        raise ValueError("decomposition expects a 3D vector.")
    return float(v[0]), float(v[1]), float(v[2])

def directional_cosines(v: np.ndarray) -> Tuple[float, float, float]:
    if v.size != 3:
        raise ValueError("dircos expects a 3D vector.")
    n = magnitude(v)
    if n == 0:
        raise ValueError("Zero vector has undefined directional cosines.")
    return float(v[0]/n), float(v[1]/n), float(v[2]/n)

def dot(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))

def cross(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return np.cross(v1, v2)

def add(*vecs: np.ndarray) -> np.ndarray:
    s = np.zeros_like(vecs[0], dtype=float)
    for v in vecs:
        s = s + v
    return s

def check_laws(v1: np.ndarray, v2: np.ndarray, v3: Optional[np.ndarray]=None, c: float=2.5, eps=1e-10) -> Dict[str, bool]:
    if v3 is None:
        v3 = np.array([0.1, -0.2, 0.3])
    antisym = np.allclose(cross(v1, v2), -cross(v2, v1), atol=eps)
    scalar_assoc = np.allclose(cross(c*v1, v2), c*cross(v1, v2), atol=eps)
    distrib1 = np.allclose(cross(v1, v2+v3), cross(v1,v2)+cross(v1,v3), atol=eps)
    distrib2 = np.allclose(cross(v1+v2, v3), cross(v1,v3)+cross(v2,v3), atol=eps)
    return {
        "antisymmetric": antisym,
        "scalar_assoc": scalar_assoc,
        "distributive_left": distrib1,
        "distributive_right": distrib2,
    }

def scalar_triple(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.dot(a, np.cross(b, c)))

def vector_triple(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return b * dot(a, c) - c * dot(a, b)

def nd_scalar_det(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    M = np.vstack([a, b, c]).T
    if M.shape[0] != 3:
        raise ValueError("nd_scalar_det expects 3D vectors.")
    return float(np.linalg.det(M))

def ortho_check(u: np.ndarray, v: np.ndarray, w: np.ndarray, eps=1e-10) -> Dict[str, Any]:
    d_uv, d_vw, d_wu = dot(u,v), dot(v,w), dot(w,u)
    return {"u·v": d_uv, "v·w": d_vw, "w·u": d_wu,
            "is_orthogonal": (abs(d_uv)<eps and abs(d_vw)<eps and abs(d_wu)<eps)}

def gram_schmidt(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    e1 = unit(v1)
    v2p = v2 - dot(v2, e1)*e1
    e2 = unit(v2p)
    v3p = v3 - dot(v3, e1)*e1 - dot(v3, e2)*e2
    e3 = unit(v3p)
    return e1, e2, e3


# ----------------------------
# Rotation / transform utils
# ----------------------------
def rot_from_euler(angles: np.ndarray, order: str="ZYX", degrees: bool=True) -> np.ndarray:
    a = np.array(angles, dtype=float)
    if degrees: a = np.deg2rad(a)
    def Rx(t): ct, st = np.cos(t), np.sin(t); return np.array([[1,0,0],[0,ct,-st],[0,st,ct]], dtype=float)
    def Ry(t): ct, st = np.cos(t), np.sin(t); return np.array([[ct,0,st],[0,1,0],[-st,0,ct]], dtype=float)
    def Rz(t): ct, st = np.cos(t), np.sin(t); return np.array([[ct,-st,0],[st,ct,0],[0,0,1]], dtype=float)
    m = {"X": Rx, "Y": Ry, "Z": Rz}
    if len(order) != 3 or any(ax not in "XYZ" for ax in order):
        raise ValueError("--order must be like ZYX, XYZ, ZXY, etc.")
    R = np.eye(3)
    for ax, ang in zip(order, a):
        R = m[ax](ang) @ R
    return R

def euler_from_rot(R: np.ndarray, order: str="ZYX", degrees: bool=True) -> np.ndarray:
    if order == "ZYX":
        if abs(R[2,0]) < 1.0:
            theta_y = -np.arcsin(R[2,0])
            theta_x = np.arctan2(R[2,1], R[2,2])
            theta_z = np.arctan2(R[1,0], R[0,0])
        else:
            theta_y = -np.pi/2 if R[2,0] <= -1 else np.pi/2
            theta_x = np.arctan2(-R[0,1], -R[0,2]); theta_z = 0.0
        ang = np.array([theta_z, theta_y, theta_x])
    elif order == "XYZ":
        if abs(R[0,2]) < 1.0:
            t_y = np.arcsin(R[0,2])
            t_x = np.arctan2(-R[1,2], R[2,2])
            t_z = np.arctan2(-R[0,1], R[0,0])
        else:
            t_y = np.pi/2 if R[0,2] >= 1 else -np.pi/2
            t_x = np.arctan2(R[2,1], R[1,1]); t_z = 0.0
        ang = np.array([t_x, t_y, t_z])
    else:
        raise NotImplementedError(f"Euler extractor for order {order} not implemented.")
    if degrees: ang = np.rad2deg(ang)
    return ang

def transform(R: np.ndarray, r_local: np.ndarray) -> np.ndarray:
    return R @ r_local


# ----------------------------
# CSV/TXT logging
# ----------------------------
def append_txt(path: str, header: str, body: str):
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + header + "\n" + "-"*len(header) + "\n")
        f.write(body.rstrip() + "\n")

def append_csv(path: str, row: Dict[str, Any]):
    ensure_dir(path)
    newfile = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newfile:
            w.writeheader()
        w.writerow(row)


# ----------------------------
# Task execution
# ----------------------------
def show_table(title: str, rows: List[Tuple[str,str]]):
    rich_table(title, rows)

def to_list_str(x: np.ndarray, fmt: str="{:.6g}") -> str:
    return "[" + ", ".join(fmt.format(v) for v in x.ravel()) + "]"

def do_task(op: str, args: argparse.Namespace) -> Dict[str, Any]:
    v1 = parse_vec_any(args.v1) if args.v1 is not None else np.array([1.0, 2.0, 3.0])
    v1 = v1.ravel()
    res: Dict[str, Any] = {"op": op}

    if op == "magnitude":
        n = float(np.linalg.norm(v1)); res["magnitude"] = n
        show_table("Magnitude", [("v1", to_list_str(v1)), ("||v1||", f"{n:.6g}")])

    elif op == "unit":
        u = v1 / np.linalg.norm(v1); res["unit"] = u.tolist()
        show_table("Unit vector", [("v1", to_list_str(v1)), ("û", to_list_str(u))])

    elif op == "natural":
        n = float(np.linalg.norm(v1)); u = v1 / n if n else v1
        res["length"] = n; res["unit"] = u.tolist()
        show_table("Natural expression r = r û", [("v1", to_list_str(v1)), ("r", f"{n:.6g}"), ("û", to_list_str(u))])

    elif op == "decompose":
        if v1.size != 3: raise ValueError("decompose expects 3D vector.")
        x, y, z = v1.tolist(); res.update({"x":x, "y":y, "z":z})
        show_table("Decomposition r = x î + y ĵ + z k̂",
                   [("v1", to_list_str(v1)), ("x", f"{x:.6g}"), ("y", f"{y:.6g}"), ("z", f"{z:.6g}")])

    elif op == "dircos":
        if v1.size != 3: raise ValueError("dircos expects 3D vector.")
        n = float(np.linalg.norm(v1)); c1,c2,c3 = v1[0]/n, v1[1]/n, v1[2]/n
        res.update({"cosα1":float(c1), "cosα2":float(c2), "cosα3":float(c3)})
        show_table("Directional cosines", [("v1", to_list_str(v1)),
                   ("cos α1", f"{c1:.6g}"), ("cos α2", f"{c2:.6g}"), ("cos α3", f"{c3:.6g}")])

    elif op == "dot":
        v2 = parse_vec_any(args.v2) if args.v2 is not None else np.array([4.0, -5.0, 6.0])
        d = float(np.dot(v1, v2)); res.update({"v2":v2.tolist(), "dot":d})
        show_table("Dot product", [("v1", to_list_str(v1)), ("v2", to_list_str(v2)), ("v1·v2", f"{d:.6g}")])

    elif op == "cross":
        v2 = parse_vec_any(args.v2) if args.v2 is not None else np.array([4.0, -5.0, 6.0])
        c = np.cross(v1, v2); res.update({"v2":v2.tolist(), "cross":c.tolist()})
        show_table("Cross product", [("v1", to_list_str(v1)), ("v2", to_list_str(v2)), ("v1×v2", to_list_str(c))])

    elif op == "add":
        v2 = parse_vec_any(args.v2) if args.v2 is not None else np.array([1.0, 1.0, 1.0])
        v3 = parse_vec_any(args.v3) if args.v3 is not None else None
        s = add(v1, v2, *( [v3] if v3 is not None else [] ))
        res.update({"v2":v2.tolist(), "v3": (v3.tolist() if v3 is not None else None), "sum": s.tolist()})
        rows=[("v1", to_list_str(v1)), ("v2", to_list_str(v2))]
        if v3 is not None: rows.append(("v3", to_list_str(v3)))
        rows.append(("v1+...", to_list_str(s)))
        show_table("Vector addition", rows)

    elif op == "laws":
        v2 = parse_vec_any(args.v2) if args.v2 is not None else np.array([-2.0, 1.0, 0.5])
        v3 = parse_vec_any(args.v3) if args.v3 is not None else np.array([0.7, -1.1, 2.0])
        chk = check_laws(v1, v2, v3); res.update({"v2":v2.tolist(), "v3":v3.tolist(), **chk})
        rows = [
            ("v1", to_list_str(v1)), ("v2", to_list_str(v2)), ("v3", to_list_str(v3)),
            ("antisymmetric", str(chk["antisymmetric"])),
            ("scalar_assoc", str(chk["scalar_assoc"])),
            ("distrib_left", str(chk["distributive_left"])),
            ("distrib_right", str(chk["distributive_right"])),
        ]
        show_table("Cross-product laws", rows)

    elif op == "triple_scalar":
        b = parse_vec_any(args.v2) if args.v2 is not None else np.array([0.0, 1.0, 0.0])
        c = parse_vec_any(args.v3) if args.v3 is not None else np.array([0.0, 0.0, 1.0])
        val = scalar_triple(v1, b, c)
        res.update({"b":b.tolist(),"c":c.tolist(),"a·(b×c)":val})
        show_table("Scalar triple product", [("a", to_list_str(v1)), ("b", to_list_str(b)),
                   ("c", to_list_str(c)), ("a·(b×c)", f"{val:.6g}")])

    elif op == "triple_vector":
        b = parse_vec_any(args.v2) if args.v2 is not None else np.array([0.0, 1.0, 0.0])
        c = parse_vec_any(args.v3) if args.v3 is not None else np.array([0.0, 0.0, 1.0])
        val = vector_triple(v1, b, c)
        res.update({"b":b.tolist(),"c":c.tolist(),"a×(b×c)":val.tolist()})
        show_table("Vector triple product (bac–cab)",
                   [("a", to_list_str(v1)), ("b", to_list_str(b)), ("c", to_list_str(c)),
                    ("a×(b×c)", to_list_str(val)), ("bac−cab", "b(a·c) − c(a·b)")])

    elif op == "nd_scalar_det":
        b = parse_vec_any(args.v2) if args.v2 is not None else np.array([0.0, 1.0, 0.0])
        c = parse_vec_any(args.v3) if args.v3 is not None else np.array([0.0, 0.0, 1.0])
        val = nd_scalar_det(v1, b, c)
        res.update({"b":b.tolist(),"c":c.tolist(),"det([a,b,c])":val})
        show_table("Generalized scalar triple (det)",
                   [("a", to_list_str(v1)), ("b", to_list_str(b)), ("c", to_list_str(c)),
                    ("det()", f"{val:.6g}")])

    elif op == "ortho_check":
        v2 = parse_vec_any(args.v2) if args.v2 is not None else np.array([0.0,1.0,0.0])
        v3 = parse_vec_any(args.v3) if args.v3 is not None else np.array([0.0,0.0,1.0])
        chk = ortho_check(v1, v2, v3); res.update({"v2":v2.tolist(),"v3":v3.tolist(), **chk})
        show_table("Orthogonality condition",
                   [("u", to_list_str(v1)), ("v", to_list_str(v2)), ("w", to_list_str(v3)),
                    ("u·v", f"{chk['u·v']:.6g}"), ("v·w", f"{chk['v·w']:.6g}"),
                    ("w·u", f"{chk['w·u']:.6g}"), ("orthogonal?", str(chk["is_orthogonal"]))])

    elif op == "gram_schmidt":
        v2 = parse_vec_any(args.v2) if args.v2 is not None else np.array([0.0,1.0,0.0])
        v3 = parse_vec_any(args.v3) if args.v3 is not None else np.array([0.0,0.0,1.0])
        e1,e2,e3 = gram_schmidt(v1, v2, v3)
        res.update({"e1":e1.tolist(),"e2":e2.tolist(),"e3":e3.tolist()})
        show_table("Orthonormal basis via Gram–Schmidt",
                   [("e1", to_list_str(e1)), ("e2", to_list_str(e2)), ("e3", to_list_str(e3))])

    elif op == "transform":
        R = parse_mat3_any(args.Rmat) if args.Rmat is not None else np.eye(3)
        gr = transform(R, v1); res.update({"R":R.flatten().tolist(),"G_r":gr.tolist()})
        show_table("Coordinate transform",
                   [("B_r", to_list_str(v1)), ("R(row-major)", to_list_str(R.flatten())),
                    ("G_r = R B_r", to_list_str(gr))])

    elif op == "rotmat":
        ang = parse_angles_any(args.euler) if args.euler is not None else np.array([30, 20, 10])
        R = rot_from_euler(ang, order=args.order, degrees=(not args.radians))
        res.update({"euler":ang.tolist(),"order":args.order,"R":R.flatten().tolist()})
        show_table("Rotation matrix from Euler",
                   [("order", args.order), ("angles", to_list_str(ang)),
                    ("R(row-major)", to_list_str(R.flatten()))])

    elif op == "euler_from_R":
        R = parse_mat3_any(args.Rmat)
        ang = euler_from_rot(R, order=args.order, degrees=(not args.radians))
        res.update({"order":args.order,"angles":ang.tolist()})
        show_table("Euler angles from R", [("order", args.order), ("Euler", to_list_str(ang))])

    elif op == "norm":
        n = magnitude(v1); res["norm"] = n
        v2 = parse_vec_any(args.v2) if args.v2 is not None else np.array([0.5, -0.4, 0.2])
        tri_ok = (magnitude(v1+v2) <= magnitude(v1) + magnitude(v2) + 1e-12)
        res.update({"v2":v2.tolist(),"triangle_ineq":tri_ok})
        show_table("Norm & linear-space sanity",
                   [("v1", to_list_str(v1)), ("||v1||", f"{n:.6g}"),
                    ("v2", to_list_str(v2)), ("||v1+v2|| ≤ ||v1||+||v2||", str(tri_ok))])

    else:
        raise ValueError(f"Unknown op: {op}")

    return res


# ----------------------------
# YAML tasks
# ----------------------------
def run_yaml(path: str, args: argparse.Namespace, out_txt: Optional[str], out_csv: Optional[str]):
    if not YAML_OK:
        sys.exit("PyYAML not installed. pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        sys.exit("YAML must contain a non-empty 'tasks' list.")
    for t in tasks:
        op = t.get("op")
        if not op:
            print("Skipping a task without 'op'.")
            continue
        args_copy = argparse.Namespace(**vars(args))  # shallow copy
        # The YAML values can be strings OR lists; we pass through and our parsers accept both.
        for key in ("v1","v2","v3","Rmat","euler","order","radians"):
            if key in t: setattr(args_copy, key, t[key])
        result = do_task(op, args_copy)
        if out_txt:
            append_txt(out_txt, f"Task: {op}", textwrap.dedent(str(result)))
        if out_csv:
            append_csv(out_csv, result)


# ----------------------------
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kinematics.py",
        description="Vector & frame operations for robotics kinematics (Jazar §1.5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--op", type=str,
                   help="Operation keyword (unit, magnitude, natural, decompose, dircos, dot, cross, add, "
                        "laws, triple_scalar, triple_vector, nd_scalar_det, ortho_check, gram_schmidt, "
                        "transform, rotmat, euler_from_R, norm). "
                        "If omitted with --from-yaml, YAML tasks drive runs.")

    # Robust, zsh-friendly inputs (nargs='+'): allow --v1 1 2 3 OR --v1 "1,2,3"
    p.add_argument("--v1", nargs="+", help='Vector 1: "1,2,3" or 1 2 3')
    p.add_argument("--v2", nargs="+", help='Vector 2: "4,-5,6" or 4 -5 6')
    p.add_argument("--v3", nargs="+", help='Vector 3: "7,8,9" or 7 8 9')

    p.add_argument("--Rmat", nargs="+", help='3x3 row-major rotation matrix: "a,..,i" or 9 numbers')
    p.add_argument("--euler", nargs="+", help='Euler angles "a,b,c" or 3 numbers')
    p.add_argument("--order", type=str, default="ZYX", help="Euler order (ZYX, XYZ supported)")
    p.add_argument("--radians", action="store_true", help="Interpret --euler in radians (default: degrees)")

    p.add_argument("--from-yaml", type=str, help="Batch tasks file, e.g., intro/in/calcs.yaml")
    p.add_argument("--out-txt", type=str, default=None, help="Append results to this text file")
    p.add_argument("--out-csv", type=str, default=None, help="Append results to this CSV file")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)

    # Batch mode
    if args.from_yaml and not args.op:
        run_yaml(args.from_yaml, args, args.out_txt, args.out_csv)
        return

    # Single op
    if not args.op:
        sys.exit("Provide --op (or use --from-yaml with tasks). See --help for supported ops.")

    result = do_task(args.op, args)

    if args.out_txt:
        append_txt(args.out_txt, f"Op: {args.op}", textwrap.dedent(str(result)))
    if args.out_csv:
        append_csv(args.out_csv, result)


if __name__ == "__main__":
    main()
