from __future__ import annotations

from src.jsjb.feedback.repository import FeedbackDatabase
from src.jsjb.retrieval.components import FeedbackScoreCache


def test_feedback_score_cache_matches_doc_aliases(tmp_path):
    FeedbackDatabase.reset_singleton()
    db_path = tmp_path / "feedback_boost.db"
    db = FeedbackDatabase(str(db_path))
    db.record_doc_feedback("doc-id", True, query="垃圾清运")
    db.record_doc_feedback("source-alpha|Policy Title", False, query="垃圾清运")
    db.close()

    cache = FeedbackScoreCache(db_path=str(db_path))

    assert cache.get_boost_for_doc({"id": "doc-id"}) == 0.05
    assert cache.get_boost_for_doc({"source": "source-alpha", "title": "Policy Title"}) == -0.05
    assert cache.get_boost_for_doc({"id": "missing", "title": "不存在"}) == 0.0
    FeedbackDatabase.reset_singleton()


def test_feedback_score_cache_caps_large_vote_effect(tmp_path):
    FeedbackDatabase.reset_singleton()
    db_path = tmp_path / "feedback_boost_cap.db"
    db = FeedbackDatabase(str(db_path))
    for _ in range(20):
        db.record_doc_feedback("popular-doc", True, query="停车")
    for _ in range(20):
        db.record_doc_feedback("bad-doc", False, query="停车")
    db.close()

    cache = FeedbackScoreCache(db_path=str(db_path))

    assert cache.get_boost_for_doc({"id": "popular-doc"}) == 0.30
    assert cache.get_boost_for_doc({"id": "bad-doc"}) == -0.30
    FeedbackDatabase.reset_singleton()
