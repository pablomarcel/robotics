# time/apis.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from .core import SolveRequest, SolveResult
from .app import MinTimeDoubleIntegrator, TwoRPathTimeScaler, TwoRParams
from .design import Planar2RGeom
from .io import write_result_payload


# -------- Programmatic convenience APIs --------

@dataclass(slots=True)
class DoubleIntegratorAPI:
    def solve(self, *, x0: float, xf: float, m: float = 1.0, F: float = 10.0,
              mu: float = 0.0, drag: float = 0.0, out_dir: str = "time/out") -> SolveResult:
        prob = MinTimeDoubleIntegrator("double_integrator", x0, xf, m, F, mu, drag)
        res = prob.run()
        write_result_payload(res.data.get("name","double_integrator"), res.data, out_dir)
        return res


@dataclass(slots=True)
class TwoRAPI:
    def line_y(self, *, y: float, x0: float, x1: float, N: int = 200,
               tau_max: Tuple[float,float]=(100,100), out_dir: str="time/out") -> SolveResult:
        qs = Planar2RGeom().path_line_y_const(y, x0, x1, n=N)
        params = TwoRParams(tau_max=tau_max)
        prob = TwoRPathTimeScaler("twoR_line", qs, params)
        res = prob.run()
        write_result_payload(res.data.get("name","twoR_line"), res.data, out_dir)
        return res
