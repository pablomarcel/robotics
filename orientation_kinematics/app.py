"""
orientation_kinematics.app (OO orchestrator)
---------------------------------
Top-level application object that wires together the service layer and CLI.

Usage
-----
# run the CLI through the app (same verbs as orientation_kinematics.cli)
python -m orientation_kinematics.app matrix-from-axis --axis 0 0 1 --phi 1.57

# programmatic use
from orientation_kinematics.app import Application
app = Application()                 # uses default singletons
svc = app.service                   # OrientationService
R = svc.euler_to_matrix([0.1,0.2,0.3], order="ZYX")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from .utils import OrientationUtils, UTILS, NumericConfig
from .io import IOManager, IO, PathConfig
from .apis import OrientationService
from .design import generate_diagram

# We reuse the OO CLI you already have
from .cli import OrientationCLI


# ----------------------------
# App configuration container
# ----------------------------

@dataclass
class AppConfig:
    """High-level knobs for wiring the app."""
    prefer_scipy: bool = True
    # You can override IO roots if you want a different working dir in tests:
    base_dir: Optional[str] = None
    in_subdir: str = "in"
    out_subdir: str = "out"


# ----------------------------
# Application orchestrator
# ----------------------------

class Application:
    """Composes utils/io/service and exposes a CLI entrypoint."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()

        # Utils backend (allows per-app numeric policy)
        utils_cfg = NumericConfig(eps=1e-12, prefer_scipy=self.config.prefer_scipy)
        self.utils: OrientationUtils = OrientationUtils(utils_cfg)

        # IO manager (reuse global IO by default, or build a local one)
        if self.config.base_dir:
            self.io: IOManager = IOManager(
                PathConfig(base_dir=PathConfig.__annotations__['base_dir'].__args__[0](self.config.base_dir),  # type: ignore
                           in_subdir=self.config.in_subdir,
                           out_subdir=self.config.out_subdir)
            )
        else:
            self.io = IO  # module singleton

        # Service layer (inject dependencies)
        self.service = OrientationService(utils=self.utils, io=self.io)

        # CLI (inject context through its own constructor)
        self.cli = OrientationCLI()  # OrientationCLI uses module-level singletons
        # If you want it to use this app's IO/Utils strictly, adapt CLI to accept a context.

    # -------- convenience API --------

    def run_cli(self, argv: Optional[List[str]] = None) -> None:
        """Dispatch to the CLI with provided argv (or sys.argv)."""
        self.cli.run(argv)

    def diagram(self, out_dir: Optional[str] = None):
        """Generate class diagrams into the configured out directory."""
        from pathlib import Path
        target = Path(out_dir) if out_dir else self.io.config.out_dir
        return generate_diagram(target)


# ----------------------------
# __main__ entrypoint
# ----------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Run the CLI through the Application object."""
    app = Application()
    app.run_cli(argv)


if __name__ == "__main__":
    main()
