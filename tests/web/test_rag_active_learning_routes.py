from __future__ import annotations

from flask import Flask

from src.jsjb.web.routes.rag_active_learning import register_rag_active_learning_routes


class DummyFeedbackDB:
    def __init__(self):
        self.reviewed = []

    def list_rag_active_learning_candidates(self, status="pending_review", limit=50):
        return [{"id": 1, "status": status, "badge_lite_score": 0.8}][:limit]

    def get_rag_active_learning_stats(self):
        return {"total_count": 1, "pending_review": 1, "by_status": {"pending_review": 1}}

    def review_rag_active_learning_candidate(
        self,
        candidate_id,
        action,
        reviewer="",
        review_label="",
        review_notes="",
    ):
        self.reviewed.append((candidate_id, action, reviewer, review_label, review_notes))
        return candidate_id == 1


class DummyLogger:
    def error(self, *args, **kwargs):
        pass


def test_rag_active_learning_routes():
    app = Flask(__name__)
    db = DummyFeedbackDB()
    register_rag_active_learning_routes(app, feedback_db=db, logger=DummyLogger())
    client = app.test_client()

    candidates = client.get("/api/rag/active-learning/candidates")
    assert candidates.status_code == 200
    assert candidates.get_json()["count"] == 1

    stats = client.get("/api/rag/active-learning/stats")
    assert stats.status_code == 200
    assert stats.get_json()["pending_review"] == 1

    review = client.post(
        "/api/rag/active-learning/review/1",
        json={"action": "approve", "reviewer": "tester", "review_label": "useful"},
    )
    assert review.status_code == 200
    assert db.reviewed[0][:4] == (1, "approve", "tester", "useful")
