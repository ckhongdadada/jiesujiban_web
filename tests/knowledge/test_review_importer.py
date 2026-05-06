from __future__ import annotations

from src.jsjb.knowledge.graph import KnowledgeGraphManager
from src.jsjb.knowledge.review_importer import import_reviewed_fact
from src.jsjb.knowledge.structured_kb import StructuredKnowledgeBase


def test_reviewed_responsible_unit_imports_to_kb_and_graph(tmp_path):
    kb_path = tmp_path / "structured_kb.json"
    graph_path = tmp_path / "knowledge_graph.json"
    kb = StructuredKnowledgeBase(str(kb_path))
    graph = KnowledgeGraphManager()

    result = import_reviewed_fact(
        {
            "fact_type": "responsible_unit",
            "fact_content": {
                "project": "朝阳路改造工程",
                "unit": "朝阳区住建委",
                "source_text": "朝阳路改造工程由朝阳区住建委负责。",
            },
        },
        feedback={"district": "朝阳区"},
        graph_manager=graph,
        graph_path=str(graph_path),
        knowledge_base=kb,
    )

    assert result["structured_kb_updated"] is True
    assert result["graph_updated"] is True
    assert result["graph_saved"] is True
    assert graph_path.exists()

    mapping = kb.get_unit_mapping("朝阳路改造工程")
    assert mapping is not None
    assert mapping.responsible_unit == "朝阳区住建委"
    assert mapping.district == "朝阳区"

    project_nodes = graph.find_by_name("Project", "朝阳路改造工程")
    org_nodes = graph.find_by_name("Organization", "朝阳区住建委")
    assert project_nodes
    assert org_nodes
    assert any(edge["type"] == "RESPONSIBLE_FOR" for edge in graph.graph.edges)


def test_reviewed_demolition_status_updates_structured_project(tmp_path):
    kb = StructuredKnowledgeBase(str(tmp_path / "structured_kb.json"))

    result = import_reviewed_fact(
        {
            "fact_type": "demolition_status",
            "fact_content": {
                "project": "南大街腾退项目",
                "demolition_status": "进行中",
                "source_text": "南大街腾退项目征收正在推进。",
            },
        },
        feedback={"district": "通州区"},
        knowledge_base=kb,
    )

    assert result["structured_kb_updated"] is True
    project = kb.get_project_status("南大街腾退项目")
    assert project is not None
    assert project.demolition_status == "进行中"
    assert project.district == "通州区"
