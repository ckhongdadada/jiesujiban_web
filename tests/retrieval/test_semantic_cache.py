from __future__ import annotations

import json

from src.jsjb.retrieval.bge_retriever import PolicyRetriever


def _write_docs(path):
    docs = [
        {
            "id": "doc-1",
            "title": "朝阳区垃圾清运案例",
            "doc_type": "案例",
            "district": "朝阳区",
            "source": "unit-test",
            "content": "朝阳区小区垃圾清运不及时，街道协调环卫单位增加清运频次。",
            "tags": ["投诉"],
        },
        {
            "id": "doc-2",
            "title": "海淀区停车治理案例",
            "doc_type": "案例",
            "district": "海淀区",
            "source": "unit-test",
            "content": "海淀区停车秩序治理，交管部门联合社区开展整治。",
            "tags": ["投诉"],
        },
    ]
    path.write_text("\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs) + "\n", encoding="utf-8")


def test_semantic_cache_exact_hit_marks_results(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_docs(corpus_path)
    retriever = PolicyRetriever(
        corpus_path=str(corpus_path),
        backend="tfidf",
        enable_query_rewrite=False,
        enable_chunking=False,
        enable_graph_augment=False,
        enable_feedback_boost=False,
        enable_reranker=False,
        enable_semantic_cache=True,
    )

    first = retriever.search("垃圾清运不及时", top_k=1, district="朝阳区")
    second = retriever.search("垃圾清运不及时", top_k=1, district="朝阳区")

    assert first
    assert second
    assert not first[0].get("cache_hit", False)
    assert second[0]["cache_hit"] is True
    assert second[0]["cache_similarity"] == 1.0
    assert retriever.describe()["semantic_cache"]["hits"] >= 1


def test_semantic_cache_respects_metadata_key(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_docs(corpus_path)
    retriever = PolicyRetriever(
        corpus_path=str(corpus_path),
        backend="tfidf",
        enable_query_rewrite=False,
        enable_chunking=False,
        enable_graph_augment=False,
        enable_feedback_boost=False,
        enable_reranker=False,
        enable_semantic_cache=True,
    )

    retriever.search("垃圾清运不及时", top_k=1, district="朝阳区")
    other_district = retriever.search("垃圾清运不及时", top_k=1, district="海淀区")

    assert other_district
    assert other_district[0].get("cache_hit") is not True
