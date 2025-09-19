# motion/tests/test_utils.py
import json
import math
import time
import types
import numpy as np
import pytest

from motion import utils


def test_timing_on_dict_adds_elapsed_ms():
    @utils.timing
    def f():
        time.sleep(0.001)  # 1 ms
        return {"ok": True}

    out = f()
    assert isinstance(out, dict)
    assert out["ok"] is True
    assert "elapsed_ms" in out and out["elapsed_ms"] >= 1.0


def test_timing_on_object_sets_attr():
    @utils.timing
    def g():
        time.sleep(0.001)
        return types.SimpleNamespace(val=42)

    obj = g()
    assert getattr(obj, "_elapsed_ms", 0) >= 1.0
    assert obj.val == 42


def test_is_rotation_matrix_true_false():
    Rz = np.array([[0, -1, 0],
                   [1,  0, 0],
                   [0,  0, 1]], dtype=float)
    assert utils.is_rotation_matrix(Rz)

    bad = np.eye(3) * 2.0
    assert not utils.is_rotation_matrix(bad)


def test_is_se3_true_false():
    T = np.eye(4)
    R = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1,  0]], dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.array([1.0, 2.0, 3.0])
    assert utils.is_se3(T)

    bad = np.eye(4)
    bad[3, 3] = 2.0
    assert not utils.is_se3(bad)


def test_numpy_json_encoder_roundtrip(tmp_path):
    payload = {
        "array": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "scalar": np.float64(3.14),
        "int": np.int64(7),
        "bool": np.bool_(True),
    }
    path = tmp_path / "payload.json"
    utils.to_json(payload, path)
    loaded = utils.from_json(path)
    # Arrays come back as lists; scalars as native Python types
    assert loaded["array"] == [[1.0, 2.0], [3.0, 4.0]]
    assert isinstance(loaded["scalar"], float) and loaded["scalar"] == pytest.approx(3.14)
    assert isinstance(loaded["int"], int) and loaded["int"] == 7
    assert isinstance(loaded["bool"], bool) and loaded["bool"] is True


def test_angle_conversions():
    assert utils.to_radians(180.0, degrees=True) == pytest.approx(math.pi)
    assert utils.to_degrees(math.pi) == pytest.approx(180.0)


def test_as_vec3_valid_and_invalid():
    v = utils.as_vec3([1, 2, 3])
    assert np.allclose(v, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        utils.as_vec3([1, 2])            # wrong length
    with pytest.raises(ValueError):
        utils.as_vec3([1, 2, float("nan")])  # non-finite


def test_almost_equal_and_clamp_and_version():
    assert utils.almost_equal(np.array([1, 2, 3.000000001]),
                              np.array([1, 2, 3]), atol=1e-8)
    assert utils.clamp(10.0, 0.0, 5.0) == 5.0
    assert isinstance(utils.version_string(), str) and len(utils.version_string()) >= 5
