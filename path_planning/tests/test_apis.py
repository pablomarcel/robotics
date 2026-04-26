import importlib.util
import numpy as np
import pytest
from path_planning.apis import poly_api, ik2r_api

def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

def _has_fastapi_httpx() -> bool:
    return _has("fastapi") and _has("httpx")

def test_poly_api_quintic_returns_coeffs():
    resp = poly_api(kind="quintic", t0=0, tf=1, q0=10, qf=45, samples=10)
    assert "coeffs" in resp and len(resp["coeffs"]) == 6
    assert resp["q"][0] == pytest.approx(10)
    assert resp["q"][-1] == pytest.approx(45)

def test_ik2r_line_basic():
    resp = ik2r_api(l1=0.25, l2=0.25, path_type="line",
                    x0=0.2, y0=0.1, x1=0.1, y1=0.2, samples=20)
    assert len(resp["th1"]) == 20
    assert len(resp["th2"]) == 20

@pytest.mark.skipif(not _has_fastapi_httpx(), reason="FastAPI/HTTPX not installed")
def test_http_fastapi_smoke():
    from fastapi.testclient import TestClient
    from path_planning.apis import get_http_app
    client = TestClient(get_http_app())
    r = client.post("/poly", json={"kind":"cubic","t0":0,"tf":1,"q0":10,"qf":45,"samples":5})
    assert r.status_code == 200 and "q" in r.json()
