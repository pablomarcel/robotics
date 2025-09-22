# Applied Dynamics (OO + TDD)

- Equations covered: 10.1–10.398 (structure ready; several canonical models implemented).
- **OO core**: `Inertia`, `FrameState`, `LagrangeEngine`, `NewtonEuler`, plus `System` models.
- **TDD**: `pytest` tests in `applied/tests/`.
- **CLI**: `python -m applied.cli <cmd>` where `<cmd>` ∈ {`pendulum`, `spherical`, `planar2r`, `absorber`, `diagram`}.

## Class diagram
```bash
pip install pylint graphviz pylint-pyreverse
python -m applied.cli diagram
# output at applied/out/applied_classes.png
