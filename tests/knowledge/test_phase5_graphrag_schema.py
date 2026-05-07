from __future__ import annotations

from src.jsjb.knowledge.graph import KnowledgeGraphManager
from src.jsjb.knowledge.review_importer import import_reviewed_fact
from src.jsjb.knowledge.schema import canonical_entity_type, canonical_relation_type, normalize_properties
from src.jsjb.knowledge.structured_kb import StructuredKnowledgeBase


def test_schema_normalizes_entity_and_relation_types():
    assert canonical_entity_type("unit") == "Organization"
    assert canonical_relation_type("responsible_for") == "RESPONSIBLE_FOR"

    props = normalize_properties("unit", "朝阳区住建委", {"district": "朝阳区", "unused": "x"})
    assert props["entity_type"] == "Organization"
    assert props["name"] == "朝阳区住建委"
    assert props["district"] == "朝阳区"
    assert "unused" not in props


def test_reviewed_project_fact_creates_stable_graph_context(tmp_path):
    kb = StructuredKnowledgeBase(str(tmp_path / "structured_kb.json"))
    graph_path = tmp_path / "knowledge_graph.json"
    graph = KnowledgeGraphManager()

    result = import_reviewed_fact(
        {
            "fact_type": "project_status",
            "fact_content": {
                "project": "朝阳路改造工程",
                "status": "已完成",
                "responsible_unit": "朝阳区住建委",
                "source_text": "朝阳路改造工程已完成，由朝阳区住建委负责。",
            },
        },
        feedback={"district": "朝阳区"},
        graph_manager=graph,
        graph_path=str(graph_path),
        knowledge_base=kb,
    )

    assert result["graph_updated"] is True
    assert result["graph_saved"] is True
    assert graph_path.exists()
    assert graph.find_by_name("Project", "朝阳路改造工程")
    assert graph.find_by_name("Organization", "朝阳区住建委")
    assert graph.find_by_name("Location", "朝阳区")
    assert any(edge["type"] == "RESPONSIBLE_FOR" for edge in graph.graph.edges)
    assert any(edge["type"] == "LOCATED_IN" for edge in graph.graph.edges)


def test_graph_manager_builds_community_summary():
    graph = KnowledgeGraphManager()
    project = graph.merge_entity(
        "Project",
        "朝阳路改造工程",
        {"district": "朝阳区", "status": "已完成", "responsible_unit": "朝阳区住建委"},
    )
    org = graph.merge_entity("Organization", "朝阳区住建委", {"district": "朝阳区"})
    graph.create_relationship(org, project, "RESPONSIBLE_FOR")

    summary = graph.build_community_summary("朝阳路改造工程由谁负责", district="朝阳区")

    assert summary["summary"]
    assert summary["node_count"] >= 1
    assert "朝阳路改造工程" in summary["summary"]
    assert "RESPONSIBLE_FOR" in summary["summary"]
