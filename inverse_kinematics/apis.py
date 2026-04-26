# inverse_kinematics/apis.py
"""
HTTP-and-Python APIs for the inverse_kinematics-kinematics package.

Two entry points:

1) InverseService — a lightweight, framework-free wrapper around InverseApp
   used by unit tests and Python callers.

2) create_rest_app() — optional FastAPI factory (only if FastAPI/Pydantic
   are installed). Not required for unit tests.

Notes for FastAPI/Pydantic v2:
- Avoid deprecated `regex=` on Field. Use validators instead.
- Use `field_validator` (v2) instead of `validator` (v1).
- Use `.model_dump()` if you need a dict from a Pydantic model instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import numpy as np

from .app import InverseApp


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

@dataclass
class InverseService:
    """
    Testable Python facade around :class:`inverse_kinematics.app.InverseApp`.

    Notes
    -----
    - Methods return plain lists/ndarrays, matching tests’ expectations.
    - No imports from web frameworks here (keeps it dead-simple to test).
    """
    app: InverseApp = InverseApp()

    # --------------------- Validation (schema-lite) ---------------------

    @staticmethod
    def validate_problem(problem: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate an IK problem dict with top-level keys: model, method, pose.

        Returns
        -------
        (ok, error) : (bool, Optional[str])
        """
        for k in ("model", "method", "pose"):
            if k not in problem:
                return False, f"Missing key '{k}' in problem."
        model, method, pose = problem["model"], problem["method"], problem["pose"]

        # Model: simple kind-based validation for now
        if not isinstance(model, dict) or "kind" not in model:
            return False, "model.kind must be provided (e.g., 'planar2r')."
        if model["kind"] == "planar2r":
            for r in ("l1", "l2"):
                if r not in model:
                    return False, f"model.{r} is required for planar2r."

        # Method
        if not isinstance(method, dict) or "method" not in method:
            return False, "method.method must be provided (e.g., 'analytic' or 'iterative')."
        if str(method["method"]).lower() not in ("analytic", "iterative"):
            return False, "method.method must be one of {'analytic','iterative'}."

        # Pose
        if not isinstance(pose, dict):
            return False, "pose must be an object."
        has_xy = ("x" in pose and "y" in pose)
        has_T = ("T" in pose)
        if not (has_xy or has_T):
            return False, "pose must include either {'x','y'} or {'T'}."

        return True, None

    # --------------------- Solvers (single pose) ----------------------

    def solve_problem(self, problem: Dict[str, Any]) -> List[List[float]]:
        """
        Solve a single IK problem of the form:
            {"model": {...}, "method": {...}, "pose": {...}}

        Returns
        -------
        list[list[float]]
            A list of solutions (each is a list of joint values).
        """
        ok, err = self.validate_problem(problem)
        if not ok:
            raise ValueError(err or "Invalid problem")
        return self.app.solve(problem)

    # Back-compat convenience (signature used by some callers/tests)
    def solve(
        self,
        model_spec: Dict[str, Any],
        method_spec: Dict[str, Any],
        pose_spec: Dict[str, Any],
        q0: Optional[Sequence[float]] = None,
    ) -> List[np.ndarray]:
        """
        Solve IK for a given model/method/pose spec (single pose).

        Returns
        -------
        list[np.ndarray]
            All solution branches for the pose.
        """
        # Build a problem dict and delegate to app.solve
        problem = {
            "model": model_spec,
            "method": method_spec if q0 is None else {**method_spec, "q0": list(q0)},
            "pose": pose_spec,
        }
        sols = self.app.solve(problem)  # list[list[float]]
        return [np.asarray(s, float) for s in sols]

    def solve_planar2r(
        self,
        l1: float,
        l2: float,
        x: float,
        y: float,
        *,
        method: str = "analytic",
        tol: float = 1e-6,
        itmax: int = 200,
        lambda_damp: float = 1e-3,
        q0: Optional[Sequence[float]] = None,
        space: str = "space",
    ) -> List[np.ndarray]:
        """
        Convenience planar-2R entry point.

        Returns
        -------
        list[np.ndarray]
        """
        sols = self.app.solve_planar2r(
            l1=l1, l2=l2, x=x, y=y,
            method=method, tol=tol, itmax=itmax,
            lambda_damp=lambda_damp, q0=q0, space=space,
        )
        return [np.asarray(s, float) for s in sols]

    # --------------------- Batch solving -------------------------------

    def solve_batch(
        self,
        model_spec: Dict[str, Any],
        method_spec: Dict[str, Any],
        poses: Iterable[Dict[str, Any]],
        q0: Optional[Sequence[float]] = None,
    ) -> List[List[np.ndarray]]:
        """
        Run IK for a set of pose specs (e.g., a path_planning).

        Returns
        -------
        list[list[np.ndarray]]
            For each pose, the list of solutions.
        """
        all_sols = self.app.solve_batch(model=model_spec, poses=list(poses), method=method_spec)
        return [[np.asarray(s, float) for s in sols_for_pose] for sols_for_pose in all_sols]

    # --------------------- Problem I/O helpers -------------------------

    def problem_from_file(self, path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Load a problem JSON file with keys: 'model', 'method', 'pose'.
        Returns (model, method, pose).
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload.get("model", {}), payload.get("method", {}), payload.get("pose", {})

    def save_solutions(self, out_path_or_name: str, solutions: List[np.ndarray]) -> str:
        """
        Save solutions JSON and return the actual file path_planning (as string).

        The file layout is a simple list-of-lists under {"solutions": ...}.
        """
        out = Path(out_path_or_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"solutions": [np.asarray(s, float).tolist() for s in solutions]}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(out)

    # --------------------- Class diagram (Mermaid) ---------------------

    def class_diagram_mermaid(self) -> str:
        """
        Generate and return a Mermaid class diagram (Markdown block as text).
        """
        out_md = self.app.out_dir / "classes.md"
        path = self.app.generate_class_diagram(out_md)
        return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Pydantic models (module-level for Pydantic v2)
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field, field_validator
except Exception:  # pragma: no cover - only triggered if FastAPI/Pydantic absent
    BaseModel = object  # type: ignore[misc,assignment]
    def field_validator(*a, **k):  # type: ignore[no-redef]
        def _wrap(fn): return fn
        return _wrap
    def Field(default, **kwargs):  # type: ignore[no-redef]
        return default  # dummy


class ProblemModel(BaseModel):
    """Generic IK problem: {'model': {...}, 'method': {...}, 'pose': {...}}"""
    model: Dict[str, Any]
    method: Dict[str, Any]
    pose: Dict[str, Any]

    @field_validator("model", "method", "pose")
    @classmethod
    def _non_empty(cls, v: Any) -> Any:
        if not isinstance(v, dict) or not v:
            raise ValueError("must be a non-empty object")
        return v


class SolveRequest(BaseModel):
    model: Dict[str, Any] = Field(..., description="Model spec (e.g., {'kind':'planar2r','l1':1,'l2':1})")
    method: Dict[str, Any] = Field(..., description="Method spec (e.g., {'method':'analytic'})")
    pose: Dict[str, Any] = Field(..., description="Pose spec ({'x','y'} or {'T'})")
    q0: Optional[List[float]] = Field(None, description="Initial guess (iterative methods)")


class Planar2RRequest(BaseModel):
    l1: float
    l2: float
    x: float
    y: float
    method: str = "analytic"          # validate by field_validator (no regex=)
    tol: float = 1e-6
    itmax: int = 200
    lambda_damp: float = 1e-3
    q0: Optional[List[float]] = None
    space: str = "space"              # validate by field_validator

    @field_validator("method")
    @classmethod
    def _method_ok(cls, v: str) -> str:
        v2 = str(v).lower()
        if v2 not in {"analytic", "iterative"}:
            raise ValueError("method must be 'analytic' or 'iterative'")
        return v2

    @field_validator("space")
    @classmethod
    def _space_ok(cls, v: str) -> str:
        v2 = str(v).lower()
        if v2 not in {"space", "body"}:
            raise ValueError("space must be 'space' or 'body'")
        return v2


class BatchSolveRequest(BaseModel):
    model: Dict[str, Any]
    method: Dict[str, Any]
    poses: List[Dict[str, Any]]
    q0: Optional[List[float]] = None


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

    # Optional: PlantUML support via py2puml
    try:
        from py2puml.py2puml import py2puml  # type: ignore
    except Exception:  # pragma: no cover - if py2puml not installed, PUML routes will 500
        py2puml = None  # type: ignore

    svc = InverseService()
    app = FastAPI(title="Inverse Kinematics API", version="0.1.0")

    # ----------------------------- helpers -----------------------------

    def _as_text(obj: Any) -> str:
        """Normalize py2puml output (str, list[str], or generator) to a single str."""
        if isinstance(obj, (str, bytes)):
            return obj.decode() if isinstance(obj, (bytes, bytearray)) else obj
        try:
            # Join any iterable of chunks into one string
            return "".join(map(str, obj))
        except TypeError:
            return str(obj)

    def _puml_text() -> str:
        if py2puml is None:
            raise RuntimeError("py2puml is not installed")
        pkg_dir = str(Path(__file__).resolve().parent)  # .../inverse_kinematics
        # py2puml often returns a generator/list of lines; normalize to str
        return _as_text(py2puml(pkg_dir, "inverse_kinematics"))

    # ----------------------------- routes ------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok", "app": svc.app.info.name, "version": svc.app.info.version}

    @app.post("/problem/validate")
    def problem_validate(problem: ProblemModel):
        ok, err = svc.validate_problem(problem.model_dump())
        return {"valid": ok, "error": err}

    @app.post("/ik/solve")
    def ik_solve(req: SolveRequest):
        ok, err = svc.validate_problem({"model": req.model, "method": req.method, "pose": req.pose})
        if not ok:
            raise HTTPException(status_code=422, detail=err)
        sols = svc.solve(req.model, req.method, req.pose, q0=req.q0)
        return {"solutions": [np.asarray(s).tolist() for s in sols]}

    @app.post("/ik/solve/planar2r")
    def ik_solve_planar2r(req: Planar2RRequest):
        sols = svc.solve_planar2r(
            l1=req.l1, l2=req.l2, x=req.x, y=req.y,
            method=req.method, tol=req.tol, itmax=req.itmax,
            lambda_damp=req.lambda_damp, q0=req.q0, space=req.space,
        )
        return {"solutions": [np.asarray(s).tolist() for s in sols]}

    @app.post("/ik/solve/batch")
    def ik_solve_batch(req: BatchSolveRequest):
        if not req.poses:
            raise HTTPException(status_code=422, detail="poses must be a non-empty list")
        # Validate using the first pose as a proxy
        ok, err = svc.validate_problem({"model": req.model, "method": req.method, "pose": req.poses[0]})
        if not ok:
            raise HTTPException(status_code=422, detail=err or "Invalid batch 'poses'.")
        sols = svc.solve_batch(req.model, req.method, req.poses, q0=req.q0)
        return {"solutions": [[[float(v) for v in s] for s in sols_for_pose] for sols_for_pose in sols]}

    @app.post("/problem/solve")
    def problem_solve(problem: ProblemModel):
        ok, err = svc.validate_problem(problem.model_dump())
        if not ok:
            raise HTTPException(status_code=422, detail=err)
        sols = svc.solve_problem(problem.model_dump())
        return {"solutions": sols}

    @app.get("/diagram/mermaid")
    def diagram_mermaid():
        return {"mermaid": svc.class_diagram_mermaid()}

    @app.get("/diagram/puml")
    def diagram_puml():
        try:
            return {"puml": _puml_text()}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PUML generation failed: {exc}")

    @app.post("/diagram/puml/save")
    def diagram_puml_save(name: str = "classes.puml"):
        """
        Write a PlantUML file under inverse_kinematics/out/{name} and return the path_planning.
        """
        try:
            txt = _puml_text()
            out = svc.app.out_dir / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(txt, encoding="utf-8")
            return {"path_planning": str(out)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PUML save failed: {exc}")

    return app


__all__ = ["InverseService", "create_rest_app",
           "ProblemModel", "SolveRequest", "Planar2RRequest", "BatchSolveRequest"]
