"""Robot Dynamics package.


Provides OOP abstractions and engines to compute robot kinematics and dynamics
covering the textbook scope (Eqs. 11.1–11.608). The code is split into
submodules to support TDD with pytest and clean architecture.
"""
from .core import Link, Joint, RobotModel, State
from .design import DHParam, DHChainBuilder
from .dynamics import DynamicsEngine, SympyLagrangeEngine, PinocchioEngine
from .apis import DynamicsAPI


__all__ = [
"Link",
"Joint",
"RobotModel",
"State",
"DHParam",
"DHChainBuilder",
"DynamicsEngine",
"SympyLagrangeEngine",
"PinocchioEngine",
"DynamicsAPI",
]