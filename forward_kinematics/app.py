# forward_kinematics/app.py
"""
High-level façade for the forward-kinematics package.

This module exposes a single OOP entry point, :class:`ForwardApp`, which
wraps the lower-level building blocks in ``core.py``, ``io.py``, and
``tools/class_diagram.py`` and adds:
    * Robot spec loading (JSON/YAML) and explicit schema validation.
    * Forward kinematics for arbitrary serial chains (DH/MDH/PoE).
    * Analytical Jacobians (space/body).
    * Preset builders (SCARA, spherical wrist types 1–3) via ``design.py``.
    * Class diagram generation via Graphviz DOT.
    * Timing and structured logging for reproducibility.

The design keeps public methods side-effect free (unless an output path_planning
is provided) so they are straightforward to unit test under pytest.

Typical usage
-------------
>>> from forward_kinematics.app import ForwardApp
>>> app = ForwardApp()
>>> chain = app.load_robot("forward_kinematics/in/example_2r.json")
>>> T = app.fk(chain, q=[0.5, 0.25])
>>> J_space = app.jacobian_space(chain, q=[0.5, 0.25])
>>> app.save_transform("forward_kinematics/out/fk.json", T)

Notes
-----
* The CLI (``forward/cli.py``) is a thin wrapper over this façade.
* JSON/YAML schema is defined in ``io.py``; validation uses ``jsonschema``.
* For class diagrams, this module defers to ``tools/class_diagram.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .utils import timed
from .core import Transform, SerialChain
from . import io as io_mod
from . import design as design_mod
from .tools.diagram import _render_dot


Number = Union[int, float]
ArrayLike = Union[Sequence[Number], np.ndarray]
PathLike = Union[str, Path]


@dataclass(frozen=True)
class AppInfo:
    """Immutable metadata about the application."""
    name: str = "forward_kinematics-kinematics"
    version: str = "0.1.0"
    homepage: str = "https://example.local/forward"  # replace when hosted


class ForwardApp:
    """
    High-level application façade for forward_kinematics kinematics workflows.

    Responsibilities
    ----------------
    * Construct :class:`SerialChain` instances from JSON/YAML specs.
    * Validate specs against the JSON Schema.
    * Provide FK and analytical Jacobians (space/body).
    * Offer convenience builders for common robots (SCARA, spherical wrists).
    * Save/Load transforms and batch computations.
    * Generate a Graphviz DOT class diagram.

    Composition
    -----------
    Delegates to:
      - :mod:`forward_kinematics.io` for parsing/validation/building serial chains.
      - :mod:`forward_kinematics.core` for Transform & kinematics math.
      - :mod:`forward_kinematics.design` for predefined robot_dynamics factories.
      - :mod:`forward_kinematics.tools.class_diagram` for DOT generation.

    All public methods use explicit inputs and return values to remain
    trivial to unit test with pytest.

    Parameters
    ----------
    in_dir : PathLike, optional
        Default input directory for specs (used by helper methods).
    out_dir : PathLike, optional
        Default output directory for results/artifacts.
    """

    def __init__(self, in_dir: Optional[PathLike] = None, out_dir: Optional[PathLike] = None):
        self.info = AppInfo()
        self.in_dir = Path(in_dir) if in_dir else Path("forward_kinematics/in")
        self.out_dir = Path(out_dir) if out_dir else Path("forward_kinematics/out")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Loading & Validation ----------

    def load_robot(self, path: PathLike, *, validate: bool = True) -> SerialChain:
        """
        Load a robot_dynamics specification (JSON or YAML) and build a :class:`SerialChain`.

        The spec format is shared between JSON and YAML. Validation uses the
        JSON Schema returned by :func:`forward_kinematics.io.robot_schema`.

        Parameters
        ----------
        path : str or Path
            File path_planning to JSON/YAML robot_dynamics specification.
        validate : bool
            Whether to validate against the schema before building.

        Returns
        -------
        SerialChain
        """
        p = Path(path)
        spec = io_mod.load_spec(p)  # JSON or YAML auto-detected
        if validate:
            io_mod.validate_spec(spec, io_mod.robot_schema())
        return io_mod.build_chain_from_spec(spec)

    def validate_file(self, path: PathLike) -> Tuple[bool, Optional[str]]:
        """
        Validate a robot_dynamics specification file against the JSON Schema.

        Returns
        -------
        (ok, error)
            ok=True if valid; otherwise error contains a human-readable message.
        """
        try:
            spec = io_mod.load_spec(path)
            io_mod.validate_spec(spec, io_mod.robot_schema())
            return True, None
        except Exception as exc:  # jsonschema.ValidationError or IO errors
            return False, str(exc)

    # ---------- Presets (Design helpers) ----------

    def preset_scara(self, l1: Number, l2: Number, d: Number = 0.0) -> SerialChain:
        """
        Build a planar SCARA (R|R|R|P) variant as a :class:`SerialChain`.

        Parameters
        ----------
        l1, l2 : float
            Link lengths (planar arm).
        d : float
            Home prismatic offset (positive along tool axis).

        Returns
        -------
        SerialChain
        """
        return design_mod.scara(l1, l2, d)

    def preset_spherical_wrist(self, wrist_type: int, d7: Number = 0.0) -> SerialChain:
        """
        Build spherical wrist types 1–3 (Roll–Pitch–Roll, Roll–Pitch–Yaw, Pitch–Yaw–Roll).

        Parameters
        ----------
        wrist_type : {1, 2, 3}
            Selects the rotation_kinematics order.
        d7 : float
            Tool-frame offset along gripper axis.

        Returns
        -------
        SerialChain
        """
        if wrist_type not in (1, 2, 3):
            raise ValueError("wrist_type must be one of {1, 2, 3}")
        return design_mod.spherical_wrist(wrist_type=wrist_type, d7=d7)

    # ---------- FK & Jacobians ----------

    @timed
    def fk(self, chain: SerialChain, q: ArrayLike) -> Transform:
        """
        Compute forward_kinematics kinematics (base-to-tool homogeneous transform).

        Parameters
        ----------
        chain : SerialChain
            The robot_dynamics model.
        q : array-like
            Joint vector (length == number of joints).

        Returns
        -------
        Transform
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        return chain.fkine(q_arr)

    def jacobian_space(self, chain: SerialChain, q: ArrayLike) -> np.ndarray:
        """
        Compute the space Jacobian J_s(q).

        Returns
        -------
        (6, n) ndarray
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        return chain.jacobian_space(q_arr)

    def jacobian_body(self, chain: SerialChain, q: ArrayLike) -> np.ndarray:
        """
        Compute the body Jacobian J_b(q).

        Returns
        -------
        (6, n) ndarray
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        return chain.jacobian_body(q_arr)

    # ---------- I/O Helpers ----------

    def save_transform(self, path: PathLike, T: Transform) -> Path:
        """
        Save a homogeneous transform as JSON matrix to disk.

        Parameters
        ----------
        path : PathLike
            Output path_planning (JSON).
        T : Transform
            Homogeneous transform to save.

        Returns
        -------
        Path  (actual file path_planning written)
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        io_mod.save_transform_json(p, T.as_matrix())
        return p

    def batch_fk(self, chain: SerialChain, q_list: Iterable[ArrayLike]) -> List[Transform]:
        """
        Compute FK for a batch of joint vectors.

        Returns
        -------
        list[Transform]
        """
        out: List[Transform] = []
        for q in q_list:
            out.append(self.fk(chain, q))
        return out

    # ---------- Class Diagram ----------

    def class_diagram_dot(self) -> str:
        """
        Build a Graphviz DOT description of the current package classes.

        Returns
        -------
        str
            DOT string that can be rendered by Graphviz.
        """
        # Introspect the forward_kinematics package to build the diagram.
        # Keep the surface stable for tests; we export only public classes.
        return render_dot(packages=("forward_kinematics.core", "forward_kinematics.io", "forward_kinematics.design", "forward_kinematics.utils"))

    def save_class_diagram(self, path: PathLike) -> Path:
        """
        Save the DOT representation of the class diagram to a file.

        Returns
        -------
        Path
        """
        dot = self.class_diagram_dot()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dot, encoding="utf-8")
        return p


__all__ = ["ForwardApp", "AppInfo"]
