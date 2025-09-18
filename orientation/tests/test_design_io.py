from pathlib import Path
import numpy as np

from orientation.design import DiagramGenerator, DiagramConfig
from orientation import core
from orientation.io import IOManager, PathConfig

def test_diagram_generation(tmp_path: Path):
    cfg = DiagramConfig(out_dir=tmp_path, modules=(core,), include_prefixes=(core.__name__,))
    gen = DiagramGenerator(cfg)
    paths = gen.generate()
    assert any(p.suffix == ".dot" for p in paths)
    assert any(p.suffix == ".mmd" for p in paths)
    for p in paths:
        assert p.exists()
        assert p.read_text()

def test_io_manager_roundtrip(tmp_path: Path):
    pcm = PathConfig(base_dir=tmp_path)
    io = IOManager(pcm)
    # JSON
    io.write_json("foo.json", {"a": 1})
    assert io.read_json("foo.json")["a"] == 1
    # CSV matrix
    M = np.eye(3)
    io.write_matrix_csv("R.csv", M)
    # Move it to in/ to read back via in_dir
    (io.config.out_dir/"R.csv").replace(io.config.in_dir/"R.csv")
    M2 = io.read_matrix_csv("R.csv")
    assert np.allclose(M, M2)
