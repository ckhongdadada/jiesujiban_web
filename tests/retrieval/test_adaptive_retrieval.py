from __future__ import annotations

import json

from src.jsjb.retrieval.bge_retriever import PolicyRetriever


def _write_docs(path, count=10):
    docs = []
    for idx in range(count):
        docs.append(
            {
                "id": f"doc-{idx}",
                "title": f"治理案例{idx}",
                "doc_type": "案例" if idx % 2 else "政策",
                "district": "朝阳区" if idx % 3 else "全市",
                "source": "unit-test",
                "content": f"小区垃圾清运、停车秩序、噪声扰民、物业服务综合治理案例 {idx}。",
                "tags": ["投诉"],
            }
        )
    path.write_text("\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs) + "\n", encoding="utf-8")


def test_adaptive_retrieval_expands_complex_query_top_k(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_docs(corpus_path, count=12)
    retriever = PolicyRetriever(
        corpus_path=str(corpus_path),
        backend="tfidf",
        enable_query_rewrite=False,
        enable_chunking=False,
        enable_graph_augment=False,
        enable_feedback_boost=False,
        enable_reranker=False,
        enable_adaptive_retrieval=True,
        adaptive_max_top_k=7,
    )

    query = (
        "小区垃圾长期无人清运，夜间施工噪声扰民，停车秩序混乱，物业多次未处理，"
        "希望明确责任单位并说明后续办理措施。"
    )
    results = retriever.search(query, top_k=3, district="", tag="投诉")

    assert results
    assert len(results) > 3
    assert results[0]["adaptive_strategy"] in {"balanced", "expanded"}
    assert results[0]["effective_top_k"] > 3
    assert results[0]["adaptive_reason"]


def test_adaptive_retrieval_can_be_disabled(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_docs(corpus_path, count=8)
    retriever = PolicyRetriever(
        corpus_path=str(corpus_path),
        backend="tfidf",
        enable_query_rewrite=False,
        enable_chunking=False,
        enable_graph_augment=False,
        enable_feedback_boost=False,
        enable_reranker=False,
        enable_adaptive_retrieval=False,
    )

    results = retriever.search("垃圾清运", top_k=3, district="朝阳区")

    assert results
    assert len(results) <= 3
    assert results[0]["adaptive_strategy"] == "standard"
