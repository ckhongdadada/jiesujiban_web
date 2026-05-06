from __future__ import annotations

from src.jsjb.knowledge.graph import InMemoryGraph
from src.jsjb.retrieval.components import GraphAugmentor
from src.jsjb.retrieval.bge_retriever import PolicyRetriever


class _GraphManager:
    def __init__(self):
        self.graph = InMemoryGraph()


def test_graph_augmentor_can_attach_runtime_graph():
    manager = _GraphManager()
    manager.graph.add_node(
        "Project",
        {
            "name": "朝阳路改造工程",
            "district": "朝阳区",
            "responsible_unit": "朝阳区住建委",
        },
    )

    augmentor = GraphAugmentor()

    assert augmentor.augment_query_terms("朝阳路改造工程", district="朝阳区") == []
    assert augmentor.attach_graph_manager(manager) is True
    assert "朝阳路改造工程" in augmentor.augment_query_terms("朝阳路改造工程", district="朝阳区")


def test_policy_retriever_attach_graph_manager_updates_existing_augmentor(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"id":"doc1","title":"测试文档","content":"朝阳路改造工程正在推进","district":"朝阳区"}\n',
        encoding="utf-8",
    )
    manager = _GraphManager()
    manager.graph.add_node(
        "Project",
        {
            "name": "朝阳路改造工程",
            "district": "朝阳区",
            "responsible_unit": "朝阳区住建委",
        },
    )

    retriever = PolicyRetriever(
        corpus_path=str(corpus_path),
        backend="tfidf",
        enable_graph_augment=True,
        enable_query_rewrite=False,
        enable_reranker=False,
    )

    assert retriever.attach_graph_manager(manager) is True
    terms = retriever._graph_augmentor.augment_query_terms("朝阳路改造工程", district="朝阳区")
    assert "朝阳路改造工程" in terms
