# acceleration_kinematics/apis.py
"""
HTTP-and-Python APIs for the acceleration_kinematics-kinematics package.

Two entry points:

1) AccelService — a lightweight, framework-free wrapper around AccelApp
   used by unit tests and Python callers.

2) create_rest_app() — optional FastAPI factory (only if FastAPI/Pydantic
   are installed). Not required for unit tests.

Design goals
------------
- Keep imports minimal so tests run without web dependencies.
- Return plain lists / ndarrays where helpful, matching test expectations.
- Provide schema-lite validation for each supported operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import json
import numpy as np

from .app import AccelApp


# ---------------------------------------------------------------------------
# Python API (service)
# ---------------------------------------------------------------------------

@dataclass
class AccelService:
    """
    Testable Python facade around :class:`acceleration_kinematics.app.AccelApp`.

    The service keeps a tiny surface that maps 1:1 to unit-testable actions:
    forward_kinematics acceleration_kinematics, inverse_kinematics acceleration_kinematics, classic α×r + ω×(ω×r),
    Euler/Quaternion helpers, and mixed-acceleration_kinematics utilities.
    """
    app: AccelApp = AccelApp()

    # --------------------- Validation helpers ---------------------

    @staticmethod
    def _require_keys(obj: Mapping[str, Any], keys: Sequence[str], ctx: str) -> Tuple[bool, Optional[str]]:
        for k in keys:
            if k not in obj:
                return False, f"Missing key '{k}' in {ctx}."
        return True, None

    @staticmethod
    def validate_problem(problem: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a generic acceleration_kinematics problem dict of the form:
            {"op": "...", "model": {...}, "payload": {...}}

        Supported ops:
          forward_kinematics, inverse_kinematics, classic, euler_alpha, quat_sb, mixed
        """
        ok, err = AccelService._require_keys(problem, ("op", "payload"), "problem")
        if not ok:
            return ok, err

        op = str(problem["op"]).lower()
        payload = problem["payload"]

        # forward_kinematics/inverse_kinematics need a model
        if op in {"forward_kinematics", "inverse_kinematics"}:
            if "model" not in problem or not isinstance(problem["model"], dict):
                return False, "model is required for 'forward_kinematics' and 'inverse_kinematics'."
            model = problem["model"]
            if model.get("kind", "").lower() != "planar2r":
                return False, "Only model.kind='planar2r' is supported in this build."
            for r in ("l1", "l2"):
                if r not in model:
                    return False, f"model.{r} is required for planar2r."

        # per-op payload validation
        if op == "forward_kinematics":
            return AccelService._require_keys(payload, ("q", "qd", "qdd"), "payload[forward_kinematics]")
        if op == "inverse_kinematics":
            return AccelService._require_keys(payload, ("q", "qd", "xdd"), "payload[inverse_kinematics]")
        if op == "classic":
            return AccelService._require_keys(payload, ("alpha", "omega", "r"), "payload[classic]")
        if op == "euler_alpha":
            return AccelService._require_keys(payload, ("angles", "rates", "accels"), "payload[euler_alpha]")
        if op == "quat_sb":
            return AccelService._require_keys(payload, ("q", "qd", "qdd"), "payload[quat_sb]")
        if op == "mixed":
            return AccelService._require_keys(payload, ("R", "omega", "alpha", "r", "vB"), "payload[mixed]")

        return False, f"Unsupported op '{op}'."

    # --------------------- Single-operation wrappers ---------------------

    def forward(self, l1: float, l2: float, q: Sequence[float], qd: Sequence[float], qdd: Sequence[float]) -> np.ndarray:
        chain = self.app.preset_planar_2r(l1, l2)
        res = self.app.forward_accel(chain, q, qd, qdd)[0]
        return np.asarray(res.xdd, float)

    def inverse(self, l1: float, l2: float, q: Sequence[float], qd: Sequence[float],
                xdd: Sequence[float], *, damping: float = 1e-8) -> np.ndarray:
        chain = self.app.preset_planar_2r(l1, l2)
        res = self.app.inverse_accel(chain, q, qd, xdd, damping=damping)[0]
        return np.asarray(res.qdd, float)

    def classic(self, alpha: Sequence[float], omega: Sequence[float], r: Sequence[float]) -> np.ndarray:
        return np.asarray(self.app.classic(alpha, omega, r), float)

    def euler_alpha(self, angles: Sequence[float], rates: Sequence[float], accels: Sequence[float]) -> np.ndarray:
        return np.asarray(self.app.euler_alpha_zyx(angles, rates, accels), float)

    def quat_sb(self, q: Sequence[float], qd: Sequence[float], qdd: Sequence[float]) -> np.ndarray:
        return np.asarray(self.app.quaternion_SB(q, qd, qdd), float)

    def mixed(self, R: Sequence[Sequence[float]], omega: Sequence[float], alpha: Sequence[float],
              r: Sequence[float], vB: Sequence[float]) -> Dict[str, List[float]]:
        aBG, aGB = self.app.mixed_G_of_B(np.asarray(R, float), omega, alpha, r, vB)
        return {"a_BG": np.asarray(aBG, float).tolist(), "a_GB": np.asarray(aGB, float).tolist()}

    # --------------------- Problem runners ---------------------

    def solve(self, problem: Dict[str, Any]) -> Any:
        ok, err = self.validate_problem(problem)
        if not ok:
            raise ValueError(err or "Invalid problem")
        return self.app.solve(problem)

    def solve_batch(self, problems: Iterable[Dict[str, Any]]) -> List[Any]:
        return [self.solve(p) for p in problems]

    # --------------------- Problem I/O helpers -------------------------

    def problem_from_file(self, path: str) -> Dict[str, Any]:
        """Load a problem JSON file (generic acceleration_kinematics payload)."""
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def save_result(self, out_path_or_name: str, result: Any) -> str:
        """
        Save a generic result JSON and return the actual file path_planning (as string).
        """
        out = Path(out_path_or_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return str(out)

    # --------------------- Class diagram (Mermaid) ---------------------

    def class_diagram_mermaid(self) -> str:
        out_md = self.app.out_dir / "classes.md"
        path = self.app.generate_class_diagram(out_md)
        return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Pydantic models (module-level for Pydantic v2; optional dependency)
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field, field_validator
except Exception:  # pragma: no cover — tests should not require FastAPI/Pydantic
    BaseModel = object  # type: ignore[misc,assignment]
    def field_validator(*a, **k):  # type: ignore[no-redef]
        def _wrap(fn): return fn
        return _wrap
    def Field(default, **kwargs):  # type: ignore[no-redef]
        return default


class ProblemModel(BaseModel):
    """Generic acceleration_kinematics problem: {'op': str, 'payload': {...}, 'model'?: {...}}"""
    op: str
    payload: Dict[str, Any]
    model: Optional[Dict[str, Any]] = None

    @field_validator("op")
    @classmethod
    def _op_ok(cls, v: str) -> str:
        v2 = str(v).lower()
        allowed = {"forward_kinematics", "inverse_kinematics", "classic", "euler_alpha", "quat_sb", "mixed"}
        if v2 not in allowed:
            raise ValueError(f"op must be one of {sorted(allowed)}")
        return v2

    @field_validator("payload")
    @classmethod
    def _payload_non_empty(cls, v: Any) -> Any:
        if not isinstance(v, dict) or not v:
            raise ValueError("payload must be a non-empty object")
        return v


class ForwardRequest(BaseModel):
    l1: float
    l2: float
    q: List[float]
    qd: List[float]
    qdd: List[float]


class InverseRequest(BaseModel):
    l1: float
    l2: float
    q: List[float]
    qd: List[float]
    xdd: List[float]
    damping: float = 1e-8


class ClassicRequest(BaseModel):
    alpha: List[float]
    omega: List[float]
    r: List[float]


class EulerAlphaRequest(BaseModel):
    angles: List[float]  # ZYX
    rates: List[float]
    accels: List[float]


class QuatSBRequest(BaseModel):
    q: List[float]
    qd: List[float]
    qdd: List[float]


class MixedRequest(BaseModel):
    R: List[List[float]]
    omega: List[float]
    alpha: List[float]
    r: List[float]
    vB: List[float]


# ---------------------------------------------------------------------------
# Optional REST API (FastAPI)
# ---------------------------------------------------------------------------

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
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI/Pydantic are required for the REST server. "
            "Install with `pip install fastapi uvicorn pydantic`."
        ) from exc

    svc = AccelService()
    app = FastAPI(title="Acceleration Kinematics API", version="0.1.0")

    # ----------------------------- routes ------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok", "app": svc.app.info.name, "version": svc.app.info.version}

    @app.post("/accel/forward")
    def accel_forward(req: ForwardRequest):
        try:
            xdd = svc.forward(req.l1, req.l2, req.q, req.qd, req.qdd)
            return {"xdd": xdd.tolist()}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/accel/inverse")
    def accel_inverse(req: InverseRequest):
        try:
            qdd = svc.inverse(req.l1, req.l2, req.q, req.qd, req.xdd, damping=req.damping)
            return {"qdd": qdd.tolist()}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/accel/classic")
    def accel_classic(req: ClassicRequest):
        try:
            a = svc.classic(req.alpha, req.omega, req.r)
            return {"a": a.tolist()}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/accel/euler-alpha")
    def accel_euler_alpha(req: EulerAlphaRequest):
        try:
            a = svc.euler_alpha(req.angles, req.rates, req.accels)
            return {"alpha": a.tolist()}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/accel/quaternion/sb")
    def accel_quaternion_sb(req: QuatSBRequest):
        try:
            S = svc.quat_sb(req.q, req.qd, req.qdd)
            return {"S_B": np.asarray(S).tolist()}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/accel/mixed")
    def accel_mixed(req: MixedRequest):
        try:
            out = svc.mixed(req.R, req.omega, req.alpha, req.r, req.vB)
            return out
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/problem/solve")
    def problem_solve(problem: ProblemModel):
        ok, err = svc.validate_problem(problem.model_dump())
        if not ok:
            raise HTTPException(status_code=422, detail=err)
        res = svc.solve(problem.model_dump())
        return {"result": res}

    return app


__all__ = [
    "AccelService",
    "create_rest_app",
    # pydantic request models
    "ProblemModel",
    "ForwardRequest",
    "InverseRequest",
    "ClassicRequest",
    "EulerAlphaRequest",
    "QuatSBRequest",
    "MixedRequest",
]
