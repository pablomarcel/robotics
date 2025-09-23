from __future__ import annotations
"""
Public programmatic + optional HTTP APIs for the path package.

- Programmatic functions: simple, dependency-free entry points wrapping PathPlannerApp.
- HTTP app: available if FastAPI is installed; provides JSON endpoints mirroring
  common operations (poly time-laws, 2R IK, rotation path). Kept minimal and
  stateless so it's easy to test and deploy.

Usage (programmatic):
    from path.apis import poly_api, ik2r_api
    resp = poly_api(kind="quintic", t0=0, tf=1, q0=10, qf=45)

Usage (HTTP, optional):
    uvicorn path.apis:get_http_app --reload
"""

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, Tuple
import numpy as np

from .app import PathPlannerApp
from .core import BoundaryConditions

# -------------------------
# Programmatic API (no deps)
# -------------------------

@dataclass
class PolyRequest:
    kind: Literal["cubic", "quintic", "septic", "lspb"]
    t0: float
    tf: float
    q0: float
    qf: float
    qd0: float = 0.0
    qdf: float = 0.0
    qdd0: float = 0.0
    qddf: float = 0.0
    vmax: Optional[float] = None
    amax: Optional[float] = None
    samples: int = 200

def poly_api(**kwargs) -> Dict[str, Any]:
    """Compute a 1D path (cubic/quintic/septic/LSPB). Returns dict with t,q,qd,qdd,coeffs."""
    req = PolyRequest(**kwargs)
    app = PathPlannerApp()
    bc = BoundaryConditions(req.t0, req.tf, req.q0, req.qf, req.qd0, req.qdf, req.qdd0, req.qddf)
    if req.kind == "lspb":
        traj = app.lspb(bc, vmax=req.vmax, amax=req.amax)
    else:
        traj = getattr(app, req.kind)(bc)
    t = np.linspace(req.t0, req.tf, req.samples)
    samp = app.sample_1d(traj, t)
    coeffs = getattr(traj, "coefficients", lambda: None)()
    return {"t": samp.t.tolist(), "q": np.asarray(samp.q).tolist(),
            "qd": np.asarray(samp.qd).tolist(), "qdd": np.asarray(samp.qdd).tolist(),
            "coeffs": None if coeffs is None else np.asarray(coeffs).tolist()}

@dataclass
class IK2RRequest:
    l1: float
    l2: float
    elbow: Literal["up","down"] = "up"
    path_type: Literal["line","circle"] = "line"
    # line:
    x0: Optional[float] = None
    y0: Optional[float] = None
    x1: Optional[float] = None
    y1: Optional[float] = None
    # circle:
    cx: Optional[float] = None
    cy: Optional[float] = None
    R: Optional[float] = None
    s0: float = 0.0
    s1: float = np.pi/2
    # time:
    t0: float = 0.0
    tf: float = 1.0
    samples: int = 200

def ik2r_api(**kwargs) -> Dict[str, Any]:
    """Planar 2R IK along a line or circular arc."""
    req = IK2RRequest(**kwargs)
    app = PathPlannerApp()
    arm = app.planar2r(req.l1, req.l2, req.elbow)
    t = np.linspace(req.t0, req.tf, req.samples)
    if req.path_type == "line":
        X = np.linspace(req.x0, req.x1, t.size)
        Y = np.linspace(req.y0, req.y1, t.size)
    else:
        s = np.linspace(req.s0, req.s1, t.size)
        X = req.cx + req.R*np.cos(s)
        Y = req.cy + req.R*np.sin(s)
    th1, th2 = arm.ik(X, Y)
    return {"t": t.tolist(), "X": X.tolist(), "Y": Y.tolist(),
            "th1": th1.tolist(), "th2": th2.tolist()}

@dataclass
class RotRequest:
    R0: np.ndarray
    Rf: np.ndarray
    samples: int = 50

def rot_api(R0, Rf, samples: int = 50) -> Dict[str, Any]:
    """Angle-axis rotation path between two rotation matrices, evenly parameterized in angle."""
    app = PathPlannerApp()
    path = app.angle_axis_path(np.asarray(R0, float), np.asarray(Rf, float))
    s = np.linspace(0, 1, samples)
    Rseq = path.R(s)
    return {"R": Rseq.tolist(), "s": s.tolist()}

# --------------------------------
# Optional FastAPI HTTP application
# --------------------------------

def get_http_app():
    """
    Return a FastAPI app if fastapi is available, else raise ImportError.
    Endpoints:
      POST /poly   -> poly_api()
      POST /ik2r   -> ik2r_api()
      POST /rot    -> rot_api()
    """
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field, confloat, conint
    except Exception as e:
        raise ImportError("FastAPI not installed. `pip install fastapi uvicorn pydantic`") from e

    app = FastAPI(title="Path Planning APIs", version="0.1.0")

    class PolyBody(BaseModel):
        kind: Literal["cubic","quintic","septic","lspb"]
        t0: float; tf: float; q0: float; qf: float
        qd0: float = 0.0; qdf: float = 0.0; qdd0: float = 0.0; qddf: float = 0.0
        vmax: float | None = None; amax: float | None = None
        samples: conint(ge=5, le=10000) = 200

    @app.post("/poly")
    def poly_endpoint(body: PolyBody):
        return poly_api(**body.dict())

    class IK2RBody(BaseModel):
        l1: float; l2: float
        elbow: Literal["up","down"] = "up"
        path_type: Literal["line","circle"] = "line"
        x0: float | None = None; y0: float | None = None
        x1: float | None = None; y1: float | None = None
        cx: float | None = None; cy: float | None = None; R: float | None = None
        s0: float = 0.0; s1: float = float(np.pi/2)
        t0: float = 0.0; tf: float = 1.0
        samples: conint(ge=5, le=10000) = 200

    @app.post("/ik2r")
    def ik2r_endpoint(body: IK2RBody):
        return ik2r_api(**body.dict())

    class RotBody(BaseModel):
        R0: list[list[float]]
        Rf: list[list[float]]
        samples: conint(ge=3, le=2000) = 50

    @app.post("/rot")
    def rot_endpoint(body: RotBody):
        return rot_api(np.array(body.R0, float), np.array(body.Rf, float), body.samples)

    return app

# Convenience for `uvicorn path.apis:get_http_app`
http_app = None
try:
    http_app = get_http_app()
except Exception:
    # FastAPI not installed; programmatic API remains available.
    pass
