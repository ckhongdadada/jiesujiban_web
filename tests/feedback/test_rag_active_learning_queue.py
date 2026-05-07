from __future__ import annotations

from src.jsjb.feedback.repository import FeedbackDatabase, get_feedback_database


def test_rag_active_learning_queue_roundtrip(tmp_path):
    FeedbackDatabase.reset_singleton()
    db = get_feedback_database(str(tmp_path / "feedback.db"))

    candidate_id = db.enqueue_rag_active_learning_candidate(
        {
            "trace_id": "trace-1",
            "query": "小区垃圾清运不及时",
            "retrieved_docs": [{"doc_id": "doc-1", "score": 0.8}],
            "top1_doc_id": "doc-1",
            "top1_score": 0.8,
            "badge_lite_score": 0.72,
            "selection_reason": ["high_retrieval_uncertainty"],
        }
    )

    candidates = db.list_rag_active_learning_candidates(limit=5)
    assert candidates[0]["id"] == candidate_id
    assert candidates[0]["retrieved_docs"][0]["doc_id"] == "doc-1"
    assert candidates[0]["selection_reason"] == ["high_retrieval_uncertainty"]

    stats = db.get_rag_active_learning_stats()
    assert stats["pending_review"] == 1

    assert db.review_rag_active_learning_candidate(candidate_id, "approve", reviewer="tester")
    assert db.list_rag_active_learning_candidates(status="approved", limit=5)[0]["id"] == candidate_id

    FeedbackDatabase.reset_singleton()
