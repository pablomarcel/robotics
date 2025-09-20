# inverse/apis.py
"""
HTTP-and-Python APIs for the inverse-kinematics package.

Two entry points:

1) InverseService — a lightweight, framework-free wrapper around InverseApp
   used by unit tests and Python callers.

2) create_rest_app() — optional FastAPI factory (only if FastAPI/Pydantic
   are installed). Not required for unit tests.
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
    Testable Python facade around :class:`inverse.app.InverseApp`.

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
        problem = {"model": model_spec, "method": method_spec, "pose": pose_spec}
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
        Run IK for a set of pose specs (e.g., a path).

        Returns
        -------
        list[list[np.ndarray]]
            For each pose, the list of solutions.
        """
        # app.solve_batch expects (model, poses, method)
        all_sols = self.app.solve_batch(model=model_spec, poses=list(poses), method=method_spec)
        # Convert list-of-lists-of-lists into list-of-lists-of-ndarrays
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
        Save solutions JSON and return the actual file path (as string).

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
        from pydantic import BaseModel, Field, validator
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI/Pydantic are required for the REST server. "
            "Install with `pip install fastapi uvicorn pydantic`."
        ) from exc

    svc = InverseService()

    # --------- Pydantic models ---------
    class ProblemModel(BaseModel):
        model: Dict[str, Any]
        method: Dict[str, Any]
        pose: Dict[str, Any]

        @validator("model", "method", "pose")
        def _non_empty(cls, v):
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
        method: str = Field("analytic", regex="^(analytic|iterative)$")
        tol: float = 1e-6
        itmax: int = 200
        lambda_damp: float = 1e-3
        q0: Optional[List[float]] = None
        space: str = Field("space", regex="^(space|body)$")

    class BatchSolveRequest(BaseModel):
        model: Dict[str, Any]
        method: Dict[str, Any]
        poses: List[Dict[str, Any]]
        q0: Optional[List[float]] = None

    # --------- FastAPI app ---------
    app = FastAPI(title="Inverse Kinematics API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "app": svc.app.info.name, "version": svc.app.info.version}

    @app.post("/problem/validate")
    def problem_validate(problem: ProblemModel):
        ok, err = svc.validate_problem(problem.dict())
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
        ok, err = svc.validate_problem(problem.dict())
        if not ok:
            raise HTTPException(status_code=422, detail=err)
        sols = svc.solve_problem(problem.dict())
        return {"solutions": sols}

    @app.get("/diagram/mermaid")
    def diagram_mermaid():
        return {"mermaid": svc.class_diagram_mermaid()}

    return app


__all__ = ["InverseService", "create_rest_app"]
