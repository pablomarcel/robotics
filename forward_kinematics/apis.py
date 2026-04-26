# forward_kinematics/apis.py
"""
HTTP-and-Python APIs for the forward-kinematics package.

This module offers two complementary entry points:

1) :class:`ForwardService`  — a small, testable Python facade around
   :class:`forward.app.ForwardApp`. This is framework-agnostic and ideal
   for unit tests (pytest) without any web stack.

2) :func:`create_rest_app`  — builds a FastAPI application *if* FastAPI
   and Pydantic are available. Endpoints cover validation, FK, Jacobians,
   presets, and class diagram export.

Usage (Python only)
-------------------
>>> from forward_kinematics.apis import ForwardService
>>> svc = ForwardService()
>>> chain = svc.load_spec({"name": "2R", "format": "dh", "joints":[...]})
>>> T = svc.forward_kinematics(chain, q=[0.3, 0.5])

Usage (HTTP, if FastAPI installed)
----------------------------------
$ uvicorn forward.apis:create_rest_app --factory --port 8000

Then:
- GET  /health
- GET  /schema
- POST /validate
- POST /fk
- POST /jacobian/space
- POST /jacobian/body
- POST /presets/scara
- POST /presets/spherical_wrist
- GET  /diagram/dot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .app import ForwardApp
from .core import SerialChain, Transform
from . import io as io_mod
from . import design as design_mod


# ------------------------------- Python API -------------------------------

@dataclass
class ForwardService:
    """
    Lightweight, testable Python service that wraps :class:`ForwardApp`.

    This contains no web-framework dependencies and is easy to mock in tests.
    """

    app: ForwardApp = ForwardApp()

    # ---- Spec I/O ----
    def load_file(self, path: str, *, validate: bool = True) -> SerialChain:
        """Load a robot_dynamics spec from JSON/YAML file and build a :class:`SerialChain`."""
        return self.app.load_robot(path, validate=validate)

    def load_spec(self, spec: Dict[str, Any], *, validate: bool = True) -> SerialChain:
        """Load a robot_dynamics spec from an in-memory dict (already parsed)."""
        if validate:
            io_mod.validate_spec(spec, io_mod.robot_schema())
        return io_mod.build_chain_from_spec(spec)

    def schema(self) -> Dict[str, Any]:
        """Return the JSON Schema used for robot_dynamics specifications."""
        return io_mod.robot_schema()

    def validate(self, spec: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a spec dict and return (ok, message)."""
        try:
            io_mod.validate_spec(spec, io_mod.robot_schema())
            return True, None
        except Exception as exc:  # jsonschema.ValidationError, KeyError, etc.
            return False, str(exc)

    # ---- Kinematics ----
    def forward_kinematics(self, chain: SerialChain, q: Sequence[float]) -> Transform:
        """Compute forward_kinematics kinematics (homogeneous transform)."""
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        return chain.fkine(q_arr)

    def jacobian_space(self, chain: SerialChain, q: Sequence[float]) -> np.ndarray:
        """Compute analytical space Jacobian J_s(q)."""
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        return chain.jacobian_space(q_arr)

    def jacobian_body(self, chain: SerialChain, q: Sequence[float]) -> np.ndarray:
        """Compute analytical body Jacobian J_b(q)."""
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        return chain.jacobian_body(q_arr)

    # ---- Presets ----
    def preset_scara(self, l1: float, l2: float, d: float = 0.0) -> SerialChain:
        """Return a SCARA preset chain."""
        return design_mod.scara(l1, l2, d)

    def preset_spherical_wrist(self, wrist_type: int, d7: float = 0.0) -> SerialChain:
        """Return a spherical wrist preset (types 1–3)."""
        return design_mod.spherical_wrist(wrist_type=wrist_type, d7=d7)

    # ---- Diagram ----
    def class_diagram_dot(self) -> str:
        """Return the Graphviz DOT for the project class diagram."""
        return self.app.class_diagram_dot()


# ------------------------------- REST  API --------------------------------
# The REST API is optional; we construct it only if FastAPI & Pydantic exist.

def create_rest_app():
    """
    Factory that returns a FastAPI app exposing the service as HTTP endpoints.

    Returns
    -------
    fastapi.FastAPI

    Raises
    ------
    ImportError
        If FastAPI or Pydantic are not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except Exception as exc:  # pragma: no cover - only hit when deps missing
        raise ImportError(
            "FastAPI/Pydantic are required for the REST server. "
            "Install with `pip install fastapi uvicorn pydantic`."
        ) from exc

    svc = ForwardService()

    class SpecModel(BaseModel):
        """Generic robot_dynamics spec (validated against jsonschema at runtime)."""
        # Accept arbitrary content; the jsonschema validator will check structure.
        __root__: Dict[str, Any]

        def dict(self, *_, **__):  # pragma: no cover - thin wrapper
            return self.__root__

    class FKRequest(BaseModel):
        spec: SpecModel = Field(..., description="Robot specification (JSON/YAML-parsed).")
        q: List[float] = Field(..., description="Joint values [q1, q2, ...]")

    class JacobianRequest(FKRequest):
        pass

    class PresetScaraRequest(BaseModel):
        l1: float
        l2: float
        d: float = 0.0
        q: List[float]

    class PresetWristRequest(BaseModel):
        wrist_type: int = Field(..., ge=1, le=3)
        d7: float = 0.0
        q: List[float]

    app = FastAPI(title="Forward Kinematics API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "app": svc.app.info.name, "version": svc.app.info.version}

    @app.get("/schema")
    def get_schema():
        return svc.schema()

    @app.post("/validate")
    def validate(spec: SpecModel):
        ok, err = svc.validate(spec.dict())
        return {"valid": ok, "error": err}

    @app.post("/fk")
    def fk(req: FKRequest):
        ok, err = svc.validate(req.spec.dict())
        if not ok:
            raise HTTPException(status_code=422, detail=err)
        chain = svc.load_spec(req.spec.dict(), validate=False)
        T = svc.forward_kinematics(chain, req.q).as_matrix()
        return {"T": T}

    @app.post("/jacobian/space")
    def jac_space(req: JacobianRequest):
        ok, err = svc.validate(req.spec.dict())
        if not ok:
            raise HTTPException(status_code=422, detail=err)
        chain = svc.load_spec(req.spec.dict(), validate=False)
        J = svc.jacobian_space(chain, req.q)
        return {"J": J}

    @app.post("/jacobian/body")
    def jac_body(req: JacobianRequest):
        ok, err = svc.validate(req.spec.dict())
        if not ok:
            raise HTTPException(status_code=422, detail=err)
        chain = svc.load_spec(req.spec.dict(), validate=False)
        J = svc.jacobian_body(chain, req.q)
        return {"J": J}

    @app.post("/presets/scara")
    def fk_scara(req: PresetScaraRequest):
        chain = svc.preset_scara(req.l1, req.l2, req.d)
        T = svc.forward_kinematics(chain, req.q).as_matrix()
        J = svc.jacobian_space(chain, req.q)
        return {"T": T, "J_space": J}

    @app.post("/presets/spherical_wrist")
    def fk_wrist(req: PresetWristRequest):
        chain = svc.preset_spherical_wrist(req.wrist_type, req.d7)
        T = svc.forward_kinematics(chain, req.q).as_matrix()
        Jb = svc.jacobian_body(chain, req.q)
        return {"T": T, "J_body": Jb}

    @app.get("/diagram/dot")
    def diagram_dot():
        return {"dot": svc.class_diagram_dot()}

    return app


__all__ = ["ForwardService", "create_rest_app"]
