from path_planning.design import DesignManager

def test_design_builds_and_writes_puml(tmp_path):
    dm = DesignManager()
    model = dm.build_model()
    puml = dm.export_plantuml(model, tmp_path/"classes.puml")
    assert puml.exists()
    # sanity: include a known class name
    text = puml.read_text()
    assert "PathPlannerApp" in text or "BoundaryConditions" in text
