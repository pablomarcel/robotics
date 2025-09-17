#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kinematics.py — Jazar §1.5 vector, frames & vector-function toolkit (CLI)

Ops (use with --op):
  unit, magnitude, natural, decompose, dircos, dot, cross, add, laws,
  triple_scalar, triple_vector, nd_scalar_det, ortho_check, gram_schmidt,
  transform, rotmat, euler_from_R, norm,
  vfunc, velocity, acceleration, jerk

Robust zsh-friendly parsing:
  Pass vectors as  --v1 1 2 3  or  --v1 "1,2,3"  or  --v1=1,2,3
  For vector functions, quote expressions: --fx "cos(w*t)".

YAML batch via --from-yaml; TXT/CSV logging via --out-txt/--out-csv.
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


# ──────────────────────────
# Utilities & robust parsing
# ──────────────────────────
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
    if arg is None:
        return None
    if isinstance(arg, np.ndarray):
        return arg.astype(float).ravel()
    if isinstance(arg, (list, tuple)):
        vals = _tokens_to_floats([str(x) for x in arg])
        return np.array(vals, dtype=float).ravel()
    s = str(arg).replace(",", " ")
    parts = [p for p in s.split() if p]
    vals = [float(p) for p in parts]
    return np.array(vals, dtype=float).ravel()

def parse_mat3_any(arg: Optional[Any]) -> Optional[np.ndarray]:
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

def parse_params_any(arg: Optional[Any]) -> Dict[sp.Symbol, float]:
    if arg is None:
        return {}
    if isinstance(arg, (list, tuple)):
        tokens = []
        for x in arg:
            x = str(x)
            tokens.extend([p for p in x.replace(",", " ").split() if p])
    else:
        tokens = [p for p in str(arg).replace(",", " ").split() if p]
    out: Dict[sp.Symbol, float] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Bad --params token '{tok}'. Use name=value.")
        name, sval = tok.split("=", 1)
        name = name.strip(); sval = sval.strip()
        out[sp.Symbol(name)] = float(sval)
    return out

def parse_t_grid(arg: Optional[Any]) -> Optional[Tuple[float,float,int]]:
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        vals = _tokens_to_floats([str(x) for x in arg])
    else:
        vals = _tokens_to_floats([str(arg)])
    if len(vals) != 3:
        raise ValueError("--t-grid needs 3 numbers: start stop steps")
    start, stop, steps_f = vals
    steps = int(round(steps_f))
    if steps < 2:
        raise ValueError("--t-grid steps must be >= 2")
    return float(start), float(stop), steps

def rich_table(title: str, rows: List[Tuple[str, str]]):
    if not RICH:
        print(f"\n{title}\n" + "-"*len(title))
        for k, v in rows:
            print(f"{k:>22s} : {v}")
        return
    t = Table(title=title, show_lines=True)
    t.add_column("Quantity", style="cyan", no_wrap=True)
    t.add_column("Value", style="white")
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)

def to_list_str(x: np.ndarray, fmt: str="{:.6g}") -> str:
    return "[" + ", ".join(fmt.format(v) for v in x.ravel()) + "]"


# ───────────────────────────────────────────────
# Core vector computations (as in previous build)
# ───────────────────────────────────────────────
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


# ─────────────────────────────
# Rotation / transform routines
# ─────────────────────────────
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


# ─────────────
# File logging
# ─────────────
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


# ───────────────────────────────────────────────
# Vector functions r(t) and derivatives (NEW)
# ───────────────────────────────────────────────
def default_fx_fy_fz():
    t = sp.symbols("t", real=True)
    w = sp.symbols("w", real=True)
    return sp.cos(w*t), sp.sin(w*t), 2*t

def get_fx_fy_fz(args) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Symbol, Dict[sp.Symbol, float]]:
    t = sp.symbols("t", real=True)
    params = parse_params_any(args.params)
    if args.fx or args.fy or args.fz:
        if not (args.fx and args.fy and args.fz):
            raise ValueError("Provide all of --fx, --fy, --fz to define r(t).")
        local = {"t": t, "pi": sp.pi, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "sqrt": sp.sqrt}
        fx = sp.sympify(args.fx, locals=local)
        fy = sp.sympify(args.fy, locals=local)
        fz = sp.sympify(args.fz, locals=local)
    else:
        fx, fy, fz = default_fx_fy_fz()
    return fx, fy, fz, t, params

def sym_sub_params(exprs: Tuple[sp.Expr,...], params: Dict[sp.Symbol,float]) -> Tuple[sp.Expr,...]:
    if not params:
        return exprs
    return tuple(sp.simplify(e.subs(params)) for e in exprs)

def lambdify_component(expr: sp.Expr, t: sp.Symbol):
    return sp.lambdify((t,), expr, "numpy")

def eval_at_t(exprs: Tuple[sp.Expr,sp.Expr,sp.Expr], t: sp.Symbol, tval: float) -> np.ndarray:
    fxs = [lambdify_component(e, t) for e in exprs]
    vals = [fx(tval) for fx in fxs]
    return np.array(vals, dtype=float).ravel()

def _ensure_series_component(comp, T: np.ndarray) -> np.ndarray:
    arr = np.array(comp, dtype=float)
    if arr.ndim == 0:                       # broadcast scalar to full grid
        arr = np.full_like(T, float(arr), dtype=float)
    return arr

def eval_on_grid(exprs: Tuple[sp.Expr,sp.Expr,sp.Expr], t: sp.Symbol, grid: Tuple[float,float,int]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    t0, t1, n = grid
    T = np.linspace(t0, t1, n)
    fxs = [lambdify_component(e, t) for e in exprs]
    X = _ensure_series_component(fxs[0](T), T)
    Y = _ensure_series_component(fxs[1](T), T)
    Z = _ensure_series_component(fxs[2](T), T)
    return T, X, Y, Z

def show_symbolic_vec(title: str, comps: Tuple[sp.Expr,sp.Expr,sp.Expr]):
    rows = [("x(t)", str(sp.simplify(comps[0]))),
            ("y(t)", str(sp.simplify(comps[1]))),
            ("z(t)", str(sp.simplify(comps[2])))]
    rich_table(title, rows)

def show_numeric_point(title: str, label: str, vec: np.ndarray, tval: float):
    rows = [(f"{label}(t={tval:g})", to_list_str(vec))]
    rich_table(title, rows)

def show_numeric_series_preview(title: str, T: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, label: str):
    rows: List[Tuple[str,str]] = []
    total = len(T)
    preview_idx = list(range(min(3, total))) + list(range(max(0, total-3), total))
    rows.append(("preview rows", f"{total} total"))
    for idx in preview_idx:
        rows.append((f"t={T[idx]:.6g}", f"{label}=[{X[idx]:.6g}, {Y[idx]:.6g}, {Z[idx]:.6g}]"))
    rich_table(title, rows)

def maybe_write_series_csv(out_csv: Optional[str], op: str, T: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    if not out_csv:
        return
    for i in range(len(T)):
        append_csv(out_csv, {"op": op, "t": float(T[i]), "x": float(X[i]), "y": float(Y[i]), "z": float(Z[i])})


# ───────────────────────
# Task execution (both)
# ───────────────────────
def show_table(title: str, rows: List[Tuple[str,str]]):
    rich_table(title, rows)

def do_task(op: str, args: argparse.Namespace) -> Dict[str, Any]:
    res: Dict[str, Any] = {"op": op}

    # Classic vector ops
    if op in {"magnitude","unit","natural","decompose","dircos","dot","cross","add",
              "laws","triple_scalar","triple_vector","nd_scalar_det",
              "ortho_check","gram_schmidt","transform","rotmat","euler_from_R","norm"}:
        v1 = parse_vec_any(args.v1) if args.v1 is not None else np.array([1.0, 2.0, 3.0]); v1 = v1.ravel()

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

        return res

    # Vector functions & derivatives
    fx, fy, fz, t, params = get_fx_fy_fz(args)
    r_vec = (fx, fy, fz)
    r_mag = sp.simplify(sp.sqrt(fx**2 + fy**2 + fz**2))
    v_vec = (sp.diff(fx, t), sp.diff(fy, t), sp.diff(fz, t))
    a_vec = (sp.diff(v_vec[0], t), sp.diff(v_vec[1], t), sp.diff(v_vec[2], t))
    j_vec = (sp.diff(a_vec[0], t), sp.diff(a_vec[1], t), sp.diff(a_vec[2], t))

    r_sub = sym_sub_params(r_vec, params)
    v_sub = sym_sub_params(v_vec, params)
    a_sub = sym_sub_params(a_vec, params)
    j_sub = sym_sub_params(j_vec, params)
    rmag_sub = sp.simplify(r_mag.subs(params))

    if op == "vfunc":
        show_symbolic_vec("r(t) — vector function", r_sub)
        rich_table("Magnitude |r(t)|", [("|r(t)|", str(rmag_sub))])
        res.update({"fx":str(r_sub[0]), "fy":str(r_sub[1]), "fz":str(r_sub[2]), "|r|":str(rmag_sub)})
    elif op == "velocity":
        show_symbolic_vec("v(t) = dr/dt", v_sub)
        res.update({"vx":str(v_sub[0]), "vy":str(v_sub[1]), "vz":str(v_sub[2])})
    elif op == "acceleration":
        show_symbolic_vec("a(t) = d²r/dt²", a_sub)
        res.update({"ax":str(a_sub[0]), "ay":str(a_sub[1]), "az":str(a_sub[2])})
    elif op == "jerk":
        show_symbolic_vec("j(t) = d³r/dt³", j_sub)
        res.update({"jx":str(j_sub[0]), "jy":str(j_sub[1]), "jz":str(j_sub[2])})
    else:
        raise ValueError(f"Unknown op: {op}")

    # Numeric evaluation
    t_scalar = args.t
    grid = parse_t_grid(args.t_grid) if args.t_grid is not None else None
    comp_map = {"vfunc": r_sub, "velocity": v_sub, "acceleration": a_sub, "jerk": j_sub}
    comps = comp_map[op]

    if t_scalar is not None:
        try:
            val = eval_at_t(comps, t, float(t_scalar))
            show_numeric_point(f"{op}(t) at t={t_scalar}", op[0], val, float(t_scalar))
            res.update({f"{op}_t": float(t_scalar), f"{op}_val": val.tolist()})
        except Exception as e:
            rich_table("Numeric evaluation error", [("message", str(e))])

    if grid is not None:
        try:
            T, X, Y, Z = eval_on_grid(comps, t, grid)
            show_numeric_series_preview(f"{op}(t) series", T, X, Y, Z, op[0])
            maybe_write_series_csv(args.out_csv, op, T, X, Y, Z)
            res.update({"t_grid": [float(grid[0]), float(grid[1]), int(grid[2])]})
        except Exception as e:
            rich_table("Series evaluation error", [("message", str(e))])

    return res


# ────────────────
# YAML batch runs
# ────────────────
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
        args_copy = argparse.Namespace(**vars(args))
        for key in ("v1","v2","v3","Rmat","euler","order","radians",
                    "fx","fy","fz","params","t","t_grid"):
            if key in t: setattr(args_copy, key, t[key])
        result = do_task(op, args_copy)
        if out_txt:
            append_txt(out_txt, f"Task: {op}", textwrap.dedent(str(result)))
        if out_csv and "t_grid" not in result:
            append_csv(out_csv, result)


# ─── CLI ───
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kinematics.py",
        description="Vector & frame operations + vector functions (Jazar §1.5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--op", type=str,
        help=("Operation keyword: unit, magnitude, natural, decompose, dircos, dot, cross, add, laws, "
              "triple_scalar, triple_vector, nd_scalar_det, ortho_check, gram_schmidt, transform, "
              "rotmat, euler_from_R, norm, vfunc, velocity, acceleration, jerk. "
              "If omitted with --from-yaml, YAML tasks drive runs."))

    # vectors (zsh-friendly)
    p.add_argument("--v1", nargs="+", help='Vector 1: "1,2,3" or 1 2 3')
    p.add_argument("--v2", nargs="+", help='Vector 2: "4,-5,6" or 4 -5 6')
    p.add_argument("--v3", nargs="+", help='Vector 3: "7,8,9" or 7 8 9')

    # rotation / euler
    p.add_argument("--Rmat", nargs="+", help='3x3 row-major rotation matrix: "a,..,i" or 9 numbers')
    p.add_argument("--euler", nargs="+", help='Euler angles "a,b,c" or 3 numbers')
    p.add_argument("--order", type=str, default="ZYX", help="Euler order (ZYX, XYZ supported)")
    p.add_argument("--radians", action="store_true", help="Interpret --euler in radians (default: degrees)")

    # vector functions
    p.add_argument("--fx", type=str, help='x(t) expression, e.g. "cos(w*t)"')
    p.add_argument("--fy", type=str, help='y(t) expression, e.g. "sin(w*t)"')
    p.add_argument("--fz", type=str, help='z(t) expression, e.g. "2*t"')
    p.add_argument("--params", nargs="+", help='Symbol values: name=value (space/comma separated)')
    p.add_argument("--t", type=float, help="Evaluate at scalar t")
    p.add_argument("--t-grid", nargs="+", help="Evaluate on grid: start stop steps")

    # I/O
    p.add_argument("--from-yaml", type=str, help="Batch tasks file, e.g., intro/in/calcs.yaml")
    p.add_argument("--out-txt", type=str, default=None, help="Append results to this text file")
    p.add_argument("--out-csv", type=str, default=None, help="Append results to this CSV file")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)

    if args.from_yaml and not args.op:
        run_yaml(args.from_yaml, args, args.out_txt, args.out_csv)
        return

    if not args.op:
        sys.exit("Provide --op (or use --from-yaml). See --help for supported ops.")

    result = do_task(args.op, args)

    if args.out_txt:
        append_txt(args.out_txt, f"Op: {args.op}", textwrap.dedent(str(result)))
    if args.out_csv and "t_grid" not in result:
        append_csv(args.out_csv, result)


if __name__ == "__main__":
    main()
