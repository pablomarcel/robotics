# Robotics

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-3D74F7.svg)](https://pablomarcel.github.io/robotics/)
[![Build & Publish Docs](https://github.com/pablomarcel/robotics/actions/workflows/pages.yml/badge.svg)](https://github.com/pablomarcel/robotics/actions/workflows/pages.yml)

Robotics is a Python-based collection of command-line engineering tools for studying, reproducing, and extending core robotics workflows in kinematics, dynamics, trajectory generation, path planning, and control.

The project is organized as a set of focused packages. Each package targets a specific robotics topic and provides a reproducible workflow through command-line interfaces, package-level examples, structured inputs, file-based outputs, and tests. The implementation is informed by standard robotics coursework and by topics from Reza N. Jazar's *Theory of Applied Robotics*, with an emphasis on transparent numerical and symbolic computation in Python.

## Documentation

Live documentation is available here:

**https://pablomarcel.github.io/robotics/**

Per-package documentation:

| Package | Documentation | Focus |
|---|---:|---|
| `acceleration_kinematics` | [docs](https://pablomarcel.github.io/robotics/acceleration_kinematics/) | Angular acceleration, rigid-body acceleration, acceleration mappings, and higher-order kinematic derivatives. |
| `angular_velocity` | [docs](https://pablomarcel.github.io/robotics/angular_velocity/) | Angular velocity vectors, frame derivatives, angular-rate mappings, and body/space representations. |
| `applied_dynamics` | [docs](https://pablomarcel.github.io/robotics/applied_dynamics/) | Applied dynamics calculations involving forces, moments, momentum, work, energy, and equations of motion. |
| `control_techniques` | [docs](https://pablomarcel.github.io/robotics/control_techniques/) | Robotics control workflows, including open-loop, closed-loop, computed-torque, and PID-style studies. |
| `forward_kinematics` | [docs](https://pablomarcel.github.io/robotics/forward_kinematics/) | Forward kinematics, frame transformations, manipulator pose calculations, and Denavit-Hartenberg workflows. |
| `inverse_kinematics` | [docs](https://pablomarcel.github.io/robotics/inverse_kinematics/) | Inverse kinematics, iterative methods, decoupling strategies, singularity checks, and pose-solving workflows. |
| `motion_planning` | [docs](https://pablomarcel.github.io/robotics/motion_planning/) | Motion-planning examples and supporting utilities for robot motion studies. |
| `orientation_kinematics` | [docs](https://pablomarcel.github.io/robotics/orientation_kinematics/) | Orientation representations, rotation matrices, Euler parameters, quaternions, and transformation utilities. |
| `path_planning` | [docs](https://pablomarcel.github.io/robotics/path_planning/) | Path planning, polynomial and non-polynomial path generation, spatial paths, and rotational paths. |
| `robot_modeling` | [docs](https://pablomarcel.github.io/robotics/robot_modeling/) | Robot structure, geometry, modeling utilities, and supporting representations. |
| `rotation_kinematics` | [docs](https://pablomarcel.github.io/robotics/rotation_kinematics/) | Rotation matrices, Euler sequences, active/passive transformations, angular-rate maps, and closed-form relations. |
| `time_optimization` | [docs](https://pablomarcel.github.io/robotics/time_optimization/) | Trajectory timing, time optimization, minimum-time motion, and bang-bang-style workflows. |
| `velocity_kinematics` | [docs](https://pablomarcel.github.io/robotics/velocity_kinematics/) | Velocity kinematics, Jacobian-based calculations, forward and inverse velocity studies. |

## Project Goals

Robotics is intended to serve as a practical computational environment for robotics study, verification, and early-stage design exploration. The main goals are:

- provide focused, testable tools for individual robotics topics;
- keep analyses reproducible through command-line execution and structured inputs;
- generate useful engineering artifacts such as JSON, CSV, HTML, and PNG outputs;
- make textbook-style calculations easier to inspect, validate, and extend;
- support both numerical and symbolic workflows where appropriate;
- maintain a modular package structure that can grow without turning into a monolithic application.

## Repository Structure

```text
acceleration_kinematics/   # Acceleration kinematics and higher-order derivatives
angular_velocity/          # Angular velocity and angular-rate mapping tools
applied_dynamics/          # Applied dynamics workflows
control_techniques/        # Robotics control methods
forward_kinematics/        # Forward pose and frame-transformation solvers
inverse_kinematics/        # Inverse kinematics workflows
motion_planning/           # Motion-planning utilities and examples
orientation_kinematics/    # Orientation representations and transformations
path_planning/             # Path design and path-planning tools
robot_modeling/            # Robot model and geometry utilities
rotation_kinematics/       # Rotation matrices, Euler sequences, and rate maps
time_optimization/         # Trajectory timing and time-optimal workflows
velocity_kinematics/       # Velocity kinematics and Jacobian calculations
```

Most packages follow a common structure:

```text
cli.py        command-line entry point
core.py       numerical or symbolic solver logic
apis.py       request/response models or public interfaces
io.py         file I/O utilities
utils.py      shared package helpers
in/           example input files
out/          generated output files
RUNS.md       reproducible command examples
docs/         package-level Sphinx documentation
tests/        package-level validation tests
```

The exact module set varies by package, but the design intent is consistent: keep the command-line interface clear, keep solver logic separate from I/O, and keep examples reproducible.

## Design Principles

### CLI-first engineering workflows

The tools are designed to run from the terminal. Important options are exposed through command-line flags, and package-specific workflows are documented in `RUNS.md` files. This keeps calculations repeatable and easy to archive.

### Reproducible inputs and outputs

Package-level `in/` folders contain example inputs. Package-level `out/` folders receive generated results. This convention makes it easier to rerun examples, compare outputs, and keep solver behavior stable across refactors.

### Focused packages

Each package targets a narrow robotics topic. This makes the code easier to test, document, and extend while allowing each tool to evolve independently.

### Transparent numerical and symbolic computation

The project uses the Python scientific-computing ecosystem for matrix operations, symbolic derivations, optimization, plotting, and control-oriented calculations. The emphasis is on readable solver logic and inspectable intermediate results.

### Testable calculations

The repository emphasizes validation through package-level tests, known examples, and repeatable command workflows. Tests are used to protect existing behavior as solver paths and documentation support are added.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/pablomarcel/robotics.git
cd robotics

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: install the repository in editable mode
pip install -e .
```

Run a package-level command:

```bash
cd rotation_kinematics
python cli.py --help
```

Many packages also support module execution from the repository root:

```bash
python -m rotation_kinematics.rot_cli --help
```

See the package-level `RUNS.md` files for tested examples.

## Example Workflow

A typical workflow is:

```bash
cd rotation_kinematics
python cli.py compose global zyx "30,-20,45" --degrees --save r_zyx.csv
python cli.py check --from-csv r_zyx.csv
```

Generated outputs are written to the package-level `out/` directory when the CLI supports file export. Other packages follow the same general pattern: choose a solver path or command, provide structured inputs or flags, run the command, and inspect generated artifacts.

## Input and Output Conventions

Common conventions across the repository:

```text
in/      example inputs such as JSON, CSV, YAML, or VCD-style data where applicable
out/     generated outputs such as JSON, CSV, PNG, or HTML files
RUNS.md  reproducible command examples for the package
```

Common output types include:

- JSON result packs for structured numerical output;
- CSV exports for matrices, trajectories, Jacobians, roots, or time histories;
- PNG figures for static plots and reports;
- interactive HTML visualizations where supported;
- text summaries suitable for terminal inspection and documentation.

## Documentation Build

The repository publishes package-level Sphinx documentation through GitHub Pages. Each package that includes a `docs/` folder can be built independently.

Example local build:

```bash
cd rotation_kinematics/docs
make html
```

or:

```bash
sphinx-build -b html rotation_kinematics/docs rotation_kinematics/docs/_build/html
```

The GitHub Pages workflow builds all available package documentation, copies each package site into the final `_site/` folder, and generates the root documentation landing page.

## Testing

Run tests for an individual package from the repository root. For example:

```bash
pytest rotation_kinematics/tests \
  --cov \
  --cov-config=rotation_kinematics/.coveragerc \
  --cov-report=term-missing
```

To run all available tests:

```bash
pytest
```

## Requirements

The project is developed against modern Python versions and the scientific Python ecosystem. See `requirements.txt` for pinned dependencies.

Typical dependencies include:

- NumPy
- SciPy
- SymPy
- Matplotlib
- Plotly
- python-control
- pytest
- Sphinx and Furo for documentation

Dependency usage varies by package. Some tools are primarily numerical, while others include symbolic derivations, plotting, or documentation helpers.

## Development Notes

When adding or modifying a package:

- keep command-line behavior clear and documented;
- keep reusable solver logic outside the CLI when possible;
- keep example inputs under `in/`;
- write generated files under `out/`;
- update `RUNS.md` with reproducible commands;
- add or update tests for new solver paths;
- keep Sphinx documentation conservative and deploy-safe;
- verify that package links in the root documentation workflow match the current folder names.

## Contributing

Contributions are welcome when they are focused, reproducible, and tested.

Before opening a pull request:

- run tests for the affected package;
- update `RUNS.md` if command behavior changed;
- document new CLI flags in `--help` text;
- include or update example input files when adding solver paths;
- verify generated outputs are written to the expected package-level `out/` folder;
- confirm package documentation builds locally when documentation files are modified.

## References and Acknowledgments

This project is informed by standard robotics coursework and references, including Reza N. Jazar's *Theory of Applied Robotics*. It also depends on the broader Python open-source scientific-computing ecosystem, including NumPy, SciPy, SymPy, Matplotlib, Plotly, python-control, pytest, Sphinx, and related tools.

## Trademark and Affiliation Notice

MATLAB® and Simulink® are registered trademarks of The MathWorks, Inc. This project is independent and is not affiliated with The MathWorks, Reza N. Jazar, or any publisher. References to textbooks or software products are for context, citation, and interoperability only.

## License

This project is released under the MIT License. See `LICENSE` for details.
