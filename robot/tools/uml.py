"""UML helper script.

Generates class diagrams using `pyreverse` (from pylint) or PlantUML if
available. Usage:

    python -m robot.tools.uml --engine pyreverse --out robot/out/uml

"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["pyreverse", "plantuml"], default="pyreverse")
    p.add_argument("--out", type=Path, default=Path("robot/out/uml"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.engine == "pyreverse":
        subprocess.check_call(["pyreverse", "-o", "png", "-p", "robot", "robot"])  # creates classes.png
        for f in Path(".").glob("classes*.png"):
            f.replace(args.out / f.name)
    else:
        print("PlantUML mode: generate .puml with pyreverse, then run plantuml.")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())