#!/usr/bin/env bash
set -euo pipefail
viztracer --exclude_files "site-packages/*" \
          --min_duration 0.0005 --max_stack_depth 20 --ignore_c_function \
          -m rotation.rot_cli "$@" -o rotation/out/trace.html --html
echo "Profile: rotation/out/trace.html"
