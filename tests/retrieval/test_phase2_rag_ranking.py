from __future__ import annotations

import json

from src.jsjb.feedback.repository import FeedbackDatabase
from src.jsjb.retrieval.bge_retriever import PolicyRetriever
from src.jsjb.retrieval.components import BM25Index, FeedbackRelevanceScorer


def _write_jsonl(path, docs):
    with open(path, "w", encoding="utf-8") as fp:
        for doc in docs:
            fp.write(json.dumps(doc, ensure_ascii=False) + "\n")


def test_bm25_index_prefers_exact_lexical_match():
    index = BM25Index([
        "垃圾清运 桶站 环境卫生 投诉办理",
        "电梯维修 物业服务 消防通道",
    ])

    scores = index.score("垃圾清运问题")

    assert scores[0] > scores[1]
    assert scores[0] > 0


def test_policy_retriever_exposes_bm25_score_breakdown(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "id": "doc-trash",
                "title": "垃圾清运案例",
                "content": "小区垃圾清运不及时，由环卫部门核实后安排桶站清运。",
                "district": "全市",
                "doc_type": "案例",
                "source": "test",
            },
            {
                "id": "doc-elevator",
                "title": "电梯维修案例",
                "content": "电梯维修由物业和维保单位处理。",
                "district": "全市",
                "doc_type": "案例",
                "source": "test",
            },
        ],
    )
    retriever = PolicyRetriever(
        corpus_path=str(corpus_path),
        backend="tfidf",
        enable_query_rewrite=False,
        enable_chunking=False,
        enable_bm25=True,
        enable_hyde=False,
        enable_reranker=False,
        enable_graph_augment=False,
        enable_feedback_boost=False,
        enable_relevance_scorer=False,
        enable_semantic_cache=False,
    )

    hits = retriever.search("垃圾清运", top_k=1)

    assert hits[0]["doc_id"] == "doc-trash"
    assert hits[0]["bm25_score"] > 0
    assert hits[0]["score_breakdown"]["bm25_score"] == hits[0]["bm25_score"]


def test_policy_retriever_can_run_bm25_as_primary_backend(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "id": "doc-parking",
                "title": "停车秩序案例",
                "content": "小区周边机动车违停严重，属地街道联合交通部门开展停车秩序整治。",
                "district": "全市",
                "doc_type": "案例",
                "source": "test",
            },
            {
                "id": "doc-water",
                "title": "积水治理案例",
                "content": "道路积水由排水部门检查雨水管线。",
                "district": "全市",
                "doc_type": "案例",
                "source": "test",
            },
        ],
    )
    retriever = PolicyRetriever(
        corpus_path=str(corpus_path),
        backend="bm25",
        enable_query_rewrite=False,
        enable_chunking=False,
        enable_bm25=True,
        enable_hyde=False,
        enable_reranker=False,
        enable_graph_augment=False,
        enable_feedback_boost=False,
        enable_relevance_scorer=False,
        enable_semantic_cache=False,
    )

    hits = retriever.search("机动车违停停车秩序", top_k=1)

    assert retriever.active_backend == "bm25"
    assert hits[0]["doc_id"] == "doc-parking"
    assert hits[0]["bm25_score"] > 0
    assert hits[0]["score_breakdown"]["bm25_score"] == hits[0]["bm25_score"]


def test_hyde_low_recall_trigger_can_recover_candidate(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "id": "doc-handling",
                "title": "generic handling document",
                "content": "办理结果通常包含现场核实、责任单位、政策依据、整改措施和回复结论",
                "district": "全市",
                "doc_type": "参考材料",
                "source": "test",
            }
        ],
    )
    retriever = PolicyRetriever(
        corpus_path=str(corpus_path),
        backend="tfidf",
        enable_query_rewrite=False,
        enable_chunking=False,
        enable_bm25=True,
        enable_hyde=True,
        hyde_trigger_threshold=999.0,
        hyde_max_queries=1,
        enable_reranker=False,
        enable_graph_augment=False,
        enable_feedback_boost=False,
        enable_relevance_scorer=False,
        enable_semantic_cache=False,
    )

    hits = retriever.search("abcxyz", top_k=1)

    assert hits
    assert hits[0]["doc_id"] == "doc-handling"
    assert hits[0]["score_breakdown"]["hyde_used"] is True
    assert hits[0]["score_breakdown"]["hyde_query_count"] == 1


def test_feedback_relevance_scorer_learns_useful_and_useless_docs(tmp_path):
    db_path = tmp_path / "feedback.db"
    FeedbackDatabase.reset_singleton()
    db = FeedbackDatabase(str(db_path))
    db.record_doc_feedback("doc-good", True, query="垃圾清运", reason="案例很匹配")
    db.record_doc_feedback("doc-bad", False, query="垃圾清运", reason="不相关")
    db.close()

    scorer = FeedbackRelevanceScorer(db_path=str(db_path), reload_interval=0)

    good = scorer.score("垃圾清运", {"id": "doc-good", "title": "垃圾清运案例"})
    bad = scorer.score("垃圾清运", {"id": "doc-bad", "title": "无关案例"})

    assert good > 0
    assert bad < 0
    assert scorer.stats()["loaded_docs"] == 2
