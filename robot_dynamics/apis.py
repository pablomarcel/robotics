from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Any
import numpy as np
from .core import RobotModel, State
from .dynamics import DynamicsEngine, SympyLagrangeEngine, PinocchioEngine


EngineKind = Literal["sympy", "pinocchio"]


@dataclass(slots=True)
class DynamicsAPI:
    """High-level façade for dynamics computations.

    Example
    -------
    api = DynamicsAPI(engine="sympy")
    res = api.run(model, State(q, qd, qdd))
    """

    engine: EngineKind = "sympy"

    def _engine(self) -> DynamicsEngine:
        if self.engine == "sympy":
            return SympyLagrangeEngine()
        return PinocchioEngine()

    def run(self, model: RobotModel, state: State, gravity: float = 9.81) -> Dict[str, Any]:
        eng = self._engine()
        out = eng.compute(model, state, gravity=gravity)
        return {"M": out.M, "C": out.C, "g": out.g, "tau": out.tau}