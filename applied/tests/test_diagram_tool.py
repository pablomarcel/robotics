# applied/tests/test_diagram_tool.py
import re
from applied.tools import diagram as dg


def test_render_dot_text_contains_graph():
    dot = dg.render_dot()
    assert "digraph AppliedDiagram" in dot
    assert "node" in dot and "edge" in dot


def test_render_plantuml_text_header_footer():
    puml = dg.render_puml()
    assert puml.strip().startswith("@startuml")
    assert puml.strip().endswith("@enduml")


def test_facade_emitters(tmp_path):
    tool = dg.DiagramTool(dg.DiagramConfig(out_dir=tmp_path))
    p1 = tool.emit_dot(out_file="classes.dot")
    p2 = tool.emit_plantuml(out_file="classes.puml")
    p3 = tool.export_model_json(out_file="classes.json")
    # paths returned as strings
    assert p1.endswith("classes.dot")
    assert p2.endswith("classes.puml")
    assert p3.endswith("classes.json")
