from __future__ import annotations

from src.jsjb.knowledge.graph import KnowledgeGraphManager
from src.jsjb.retrieval.components import GraphAugmentor


def _build_graph() -> KnowledgeGraphManager:
    graph = KnowledgeGraphManager()
    project = graph.merge_entity(
        "Project",
        "朝阳路改造工程",
        {
            "district": "朝阳区",
            "status": "已完成",
            "responsible_unit": "朝阳区住建委",
            "description": "朝阳路改造工程已完工通车。",
        },
    )
    org = graph.merge_entity("Organization", "朝阳区住建委", {"district": "朝阳区"})
    location = graph.merge_entity("Location", "朝阳区", {"district": "朝阳区", "type": "区县"})
    graph.create_relationship(org, project, "RESPONSIBLE_FOR")
    graph.create_relationship(project, location, "LOCATED_IN")
    return graph


def test_graph_augmentor_scores_documents_with_graph_entities():
    graph = _build_graph()
    augmentor = GraphAugmentor(graph)
    doc = {
        "title": "道路办理案例",
        "content": "朝阳路改造工程已完工通车，后续由朝阳区住建委做好维护。",
        "district": "朝阳区",
        "unit": "朝阳区住建委",
    }

    score, entities = augmentor.graph_score_document(
        doc,
        "朝阳路改造工程什么时候通车",
        district="朝阳区",
        unit="朝阳区住建委",
    )

    assert score > 0
    assert "朝阳路改造工程" in entities


def test_graph_augmentor_returns_community_summary_hit():
    graph = _build_graph()
    augmentor = GraphAugmentor(graph)

    hits = augmentor.query_related_facts("朝阳路改造工程由谁负责", district="朝阳区", limit=4)

    assert hits
    assert any(hit["source"] == "knowledge_graph_community" for hit in hits)
    community = next(hit for hit in hits if hit["source"] == "knowledge_graph_community")
    assert "RESPONSIBLE_FOR" in community["content"]
    assert community["graph_community"]["relation_count"] >= 1


def test_graph_augmentor_expands_query_terms_from_manager_search():
    graph = _build_graph()
    augmentor = GraphAugmentor(graph)

    terms = augmentor.augment_query_terms("朝阳路改造工程进展", district="朝阳区")

    assert "朝阳路改造工程" in terms
    assert "朝阳区住建委" in terms
