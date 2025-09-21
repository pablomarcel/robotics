from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import pytest

from acceleration import io as accel_io

def test_schemas_sanity():
    assert "type" in accel_io.problem_schema()
    assert "type" in accel_io.chain_schema()
    assert "type" in accel_io.model_schema()

def test_validate_problem_minimal(io_dirs):
    # A tiny exemplar problem for acceleration: “planar_2r ee acceleration”
    problem = {
        "model": {"kind": "planar2r", "l1": 0.7, "l2": 0.9},
        "method": {"op": "forward_accel"},  # free-form op (module supports more)
        "query": {
            "frame": "ee",
            "state": {"q": [0.2, -0.4], "qd": [0.3, 0.1], "qdd": [0.0, 0.2]},
            "backend": {"impl": "numpy"}
        }
    }
    # We validate at least the chain part (if your problem validator separates concerns)
    accel_io.validate_high_level_model(problem["model"])

def test_build_chain_from_model_high_level():
    model = {"kind": "planar2r", "l1": 0.7, "l2": 0.9}
    chain = accel_io.build_chain_from_model(model, validate=True)
    # basic properties
    assert hasattr(chain, "links") or hasattr(chain, "n")
