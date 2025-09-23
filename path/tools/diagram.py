"""
Generate UML class diagram.

Prefers pyreverse (from pylint) to emit PlantUML or Graphviz dot.
Usage:
    python -m path.tools.diagram --fmt png --out path/out/diagram
"""
from __future__ import annotations
import argparse, subprocess, sys, shutil, pathlib

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="png", choices=["png","svg","pdf"])
    ap.add_argument("--out", default="path/out/diagram")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out).with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("pyreverse"):
        # produce dot, then render via dot if available
        subprocess.check_call(["pyreverse","-o","dot","-p","path_pkg","path"])
        dot = pathlib.Path("classes_path_pkg.dot")
        if dot.exists() and shutil.which("dot"):
            subprocess.check_call(["dot","-T"+args.fmt,"-o", str(out.with_suffix("."+args.fmt)), str(dot)])
            print("Diagram written to", out.with_suffix("."+args.fmt))
        else:
            print("pyreverse ran; dot file is in cwd. Install Graphviz 'dot' to render.")
    else:
        print("Install 'pylint' to get pyreverse (pip install pylint)")

if __name__ == "__main__":
    main()
