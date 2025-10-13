# motion/cli.py
"""
Command-line interface for Motion Kinematics.

- Pure argparse, no external CLI deps.
- Subcommands map 1:1 to APIs.
- File-driven `run --file` supports key aliases and auto-infers op when needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .apis import APIs


# ------------------------------ constants -------------------------------------
DEFAULT_IN = Path("motion/in")
DEFAULT_OUT = Path("motion/out")
VALID_OPS = ("rotation", "screw", "plucker", "lines", "plane-dist", "fk")


# ------------------------------ tiny utils ------------------------------------
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _parse_vec3(csv: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in csv.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated numbers")
    return (parts[0], parts[1], parts[2])


def _parse_dh(csv: str) -> Tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in csv.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("DH expects 4 comma-separated numbers: a,alpha,d,theta")
    return (parts[0], parts[1], parts[2], parts[3])


def _np_to_lists(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_np_to_lists(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _np_to_lists(v) for k, v in obj.items()}
    return obj


def _dump_json(payload: Dict[str, Any], out_path: Path | None) -> None:
    if out_path is None:
        name = payload.get("meta", {}).get("op", "result")
        out_path = DEFAULT_OUT / f"{name}.json"
    _ensure_dir(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_np_to_lists(payload), f, indent=2)
    print(str(out_path))


# ------------------------------ API dispatchers -------------------------------
def cmd_rotation(args: argparse.Namespace) -> Dict[str, Any]:
    api = APIs()
    result = api.rotation_axis_angle(args.axis, args.angle, degrees=args.degrees)
    result["meta"] = {**result.get("meta", {}), "op": "rotation"}
    return result


def cmd_screw(args: argparse.Namespace) -> Dict[str, Any]:
    api = APIs()
    result = api.screw_motion(args.axis, args.s, args.pitch, args.phi, degrees=args.degrees)
    result["meta"] = {**result.get("meta", {}), "op": "screw"}
    return result


def cmd_plucker(args: argparse.Namespace) -> Dict[str, Any]:
    api = APIs()
    result = api.plucker_from_points(args.p1, args.p2)
    result["meta"] = {**result.get("meta", {}), "op": "plucker"}
    return result


def cmd_lines(args: argparse.Namespace) -> Dict[str, Any]:
    api = APIs()
    result = api.plucker_angle_distance(args.a1, args.a2, args.b1, args.b2)
    result["meta"] = {**result.get("meta", {}), "op": "lines"}
    return result


def cmd_plane_dist(args: argparse.Namespace) -> Dict[str, Any]:
    api = APIs()
    result = api.plane_point_distance(args.point, args.normal, s=args.s, signed=not args.unsigned)
    result["meta"] = {**result.get("meta", {}), "op": "plane-dist"}
    return result


def cmd_fk(args: argparse.Namespace) -> Dict[str, Any]:
    api = APIs()
    result = api.forward_kinematics(args.dh)
    result["meta"] = {**result.get("meta", {}), "op": "fk"}
    return result


# ------------------------------ run helpers -----------------------------------
def _csv3(params: Dict[str, Any], key: str) -> str:
    v = params.get(key)
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return ",".join(str(x) for x in v)
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",")]
        if len(parts) == 3:
            return v
    raise SystemExit(f"Missing/invalid '{key}' in params (got keys={sorted(params.keys())})")


def _csv3_any(params: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = params.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return ",".join(str(x) for x in v)
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",")]
            if len(parts) == 3:
                return v
    raise SystemExit(f"Missing/invalid one of {keys!r} in params (got keys={sorted(params.keys())})")


def _deg_flag(params: Dict[str, Any]) -> List[str]:
    return ["--degrees"] if params.get("degrees") else []


def _fk_args(params: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    dh = params.get("dh")
    if not isinstance(dh, list):
        raise SystemExit("fk expects 'params': { 'dh': [[a,alpha,d,theta], ...] }")
    for row in dh:
        if not (isinstance(row, (list, tuple)) and len(row) == 4):
            raise SystemExit("Each DH row must have 4 numbers")
        out += ["--dh", ",".join(str(float(x)) for x in row)]
    return out


def _infer_op(params: Dict[str, Any]) -> str:
    keys = set(params.keys())
    if {"p1", "p2"} & keys:
        return "plucker"
    if {"a1", "a2", "b1", "b2"} <= keys:
        return "lines"
    if {"point", "normal"} <= keys:
        return "plane-dist"
    if "dh" in keys:
        return "fk"
    # screw vs rotation
    if (("s" in keys or "point" in keys) and (("u" in keys) or ("axis" in keys)) and (("phi" in keys) or ("angle" in keys))):
        return "screw"
    if (("u" in keys or "axis" in keys) and ("phi" in keys or "angle" in keys)):
        return "rotation"
    # default fail
    raise SystemExit(
        f"Could not infer 'op' from params keys={sorted(keys)}. "
        f"Valid ops: {', '.join(VALID_OPS)}"
    )


def _build_argv_for_op(op: str, params: Dict[str, Any]) -> List[str]:
    if op == "rotation":
        return [
            "rotation",
            "--axis",
            _csv3_any(params, ["axis", "u"]),
            "--angle",
            str(params.get("angle", params.get("phi", 0.0))),
        ] + _deg_flag(params)

    if op == "screw":
        return [
            "screw",
            "--axis",
            _csv3_any(params, ["u", "axis"]),
            "--s",
            _csv3_any(params, ["s", "point"]),
            "--pitch",
            str(params.get("h", params.get("pitch", 0.0))),
            "--phi",
            str(params.get("phi", params.get("angle", 0.0))),
        ] + _deg_flag(params)

    if op == "plucker":
        return ["plucker", "--p1", _csv3(params, "p1"), "--p2", _csv3(params, "p2")]

    if op == "lines":
        return [
            "lines",
            "--a1",
            _csv3(params, "a1"),
            "--a2",
            _csv3(params, "a2"),
            "--b1",
            _csv3(params, "b1"),
            "--b2",
            _csv3(params, "b2"),
        ]

    if op == "plane-dist":
        argv = [
            "plane-dist",
            "--point",
            _csv3_any(params, ["point"]),
            "--normal",
            _csv3_any(params, ["normal"]),
            "--s",
            str(params.get("s", 0.0)),
        ]
        if params.get("unsigned"):
            argv.append("--unsigned")
        return argv

    if op == "fk":
        return ["fk", *_fk_args(params)]

    raise SystemExit(f"Unknown op {op!r}. Valid ops: {', '.join(VALID_OPS)}")


# ------------------------------ 'run' subcommand -------------------------------
def cmd_run(args: argparse.Namespace) -> Dict[str, Any]:
    with Path(args.file).open("r", encoding="utf-8") as f:
        job = json.load(f)

    op_in = job.get("op")
    params = job.get("params", {})

    parser = build_parser()

    def _try(op_label: str) -> Dict[str, Any]:
        """Build argv for a given op label, parse, dispatch."""
        argv = _build_argv_for_op(op_label, params)
        parsed = parser.parse_args(argv)
        result = parsed.func(parsed)
        result["meta"] = {**result.get("meta", {}), "op": op_label, "source": str(args.file)}
        return result

    # 1) If op is provided, try it first (with good error messages)
    if isinstance(op_in, str):
        op_in = op_in.strip()
        if op_in in VALID_OPS:
            try:
                return _try(op_in)
            except SystemExit as e:
                # Fall back to inference with a clearer hint
                inferred = _infer_op(params)
                if inferred != op_in:
                    return _try(inferred)
                # If inference equals provided op, re-raise with context
                raise SystemExit(f"[run] Provided op '{op_in}' failed to validate: {e}") from None
        # If provided but invalid, we’ll infer.
    # 2) No/invalid op: infer from params.
    inferred = _infer_op(params)
    return _try(inferred)


# ------------------------------ parser ----------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="motion.cli", description="Motion Kinematics CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # rotation
    pr = sub.add_parser("rotation", help="Axis–angle rotation (Rodrigues)")
    pr.add_argument("--axis", type=_parse_vec3, required=True, help="ax,ay,az")
    pr.add_argument("--angle", type=float, required=True, help="angle (radians by default)")
    pr.add_argument("--degrees", action="store_true", help="interpret angle in degrees")
    pr.add_argument("--out", type=Path, help="output JSON path (default motion/out/rotation.json)")
    pr.set_defaults(func=cmd_rotation)

    # screw
    ps = sub.add_parser("screw", help="General screw motion (SE(3))")
    ps.add_argument("--axis", type=_parse_vec3, required=True, help="ux,uy,uz (unit or not)")
    ps.add_argument("--s", type=_parse_vec3, required=True, help="location vector s (any point on axis)")
    ps.add_argument("--pitch", type=float, required=True, help="pitch h")
    ps.add_argument("--phi", type=float, required=True, help="rotation angle φ (rad by default)")
    ps.add_argument("--degrees", action="store_true", help="interpret φ in degrees")
    ps.add_argument("--out", type=Path, help="output JSON path (default motion/out/screw.json)")
    ps.set_defaults(func=cmd_screw)

    # plucker from two points
    pp = sub.add_parser("plucker", help="Plücker line from two points")
    pp.add_argument("--p1", type=_parse_vec3, required=True)
    pp.add_argument("--p2", type=_parse_vec3, required=True)
    pp.add_argument("--out", type=Path, help="output JSON path (default motion/out/plucker.json)")
    pp.set_defaults(func=cmd_plucker)

    # angle & distance between lines
    pl = sub.add_parser("lines", help="Angle and distance between two lines via Plücker coords")
    pl.add_argument("--a1", type=_parse_vec3, required=True)
    pl.add_argument("--a2", type=_parse_vec3, required=True)
    pl.add_argument("--b1", type=_parse_vec3, required=True)
    pl.add_argument("--b2", type=_parse_vec3, required=True)
    pl.add_argument("--out", type=Path, help="output JSON path (default motion/out/lines.json)")
    pl.set_defaults(func=cmd_lines)

    # plane distance
    ppd = sub.add_parser("plane-dist", help="Distance from point to plane")
    ppd.add_argument("--point", type=_parse_vec3, required=True)
    ppd.add_argument("--normal", type=_parse_vec3, required=True, help="plane normal (need not be unit)")
    ppd.add_argument("--s", type=float, default=0.0, help="plane offset: n̂·x = s (default 0)")
    ppd.add_argument("--unsigned", action="store_true", help="report absolute distance")
    ppd.add_argument("--out", type=Path, help="output JSON path (default motion/out/plane-dist.json)")
    ppd.set_defaults(func=cmd_plane_dist)

    # forward kinematics (DH)
    pfk = sub.add_parser("fk", help="Forward kinematics with DH rows: a,alpha,d,theta")
    pfk.add_argument("--dh", type=_parse_dh, required=True, action="append",
                     help="DH row (repeatable). Example: --dh 0.1,0.0,0.2,1.57")
    pfk.add_argument("--out", type=Path, help="output JSON path (default motion/out/fk.json)")
    pfk.set_defaults(func=cmd_fk)

    # file-driven
    prun = sub.add_parser("run", help="Run from a JSON job file")
    prun.add_argument("--file", type=Path, required=True, help="path to motion/in/job.json")
    prun.add_argument("--out", type=Path, help="output JSON path (default based on op)")
    prun.set_defaults(func=cmd_run)

    return p


# ---------------------------------- main --------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    _dump_json(result, args.out)


if __name__ == "__main__":
    main()
