from __future__ import annotations

import json

from src.jsjb.retrieval.bge_retriever import PolicyRetriever


def test_parent_child_retrieval_returns_child_and_parent_context(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    doc = {
        "id": "doc-parent-1",
        "title": "小区综合治理案例",
        "doc_type": "案例",
        "district": "朝阳区",
        "source": "unit-test",
        "content": (
            "第一部分介绍小区环境卫生整治，重点是垃圾桶站值守和清运频次优化。"
            "第二部分介绍电梯故障处置，物业已联系维保单位排查主板问题并安排临时值守。"
            "第三部分介绍停车秩序治理，社区联合交管部门开展宣传引导。"
        ),
        "tags": ["投诉"],
    }
    corpus_path.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")

    retriever = PolicyRetriever(
        corpus_path=str(corpus_path),
        backend="tfidf",
        enable_query_rewrite=False,
        enable_chunking=True,
        enable_parent_child=True,
        chunk_size=100,
        chunk_overlap=0,
        enable_graph_augment=False,
        enable_feedback_boost=False,
        enable_reranker=False,
    )

    results = retriever.search("电梯故障维保主板问题", top_k=1, district="朝阳区")

    assert results
    hit = results[0]
    assert hit["doc_id"] == "doc-parent-1"
    assert hit["retrieval_granularity"] == "parent_child"
    assert hit["child_chunk_id"]
    assert "电梯故障" in hit["child_snippet"]
    assert "电梯故障" in hit["parent_context"]
    assert hit["full_content"] == doc["content"]
