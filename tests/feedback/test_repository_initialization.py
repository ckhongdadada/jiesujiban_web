from __future__ import annotations

from src.jsjb.feedback import repository
from src.jsjb.feedback.repository import FeedbackDatabase


def test_feedback_database_no_arg_uses_default_path(monkeypatch, tmp_path):
    FeedbackDatabase.reset_singleton()
    db_path = tmp_path / "default_feedback.db"
    monkeypatch.setattr(repository, "get_feedback_db_path", lambda: db_path)

    db = FeedbackDatabase()
    feedback_id = db.add_feedback(
        {
            "tag": "投诉",
            "title": "测试反馈",
            "body": "测试内容",
            "reply": "测试回复",
            "unit": "测试单位",
            "district": "朝阳区",
            "is_helpful": True,
        }
    )

    assert feedback_id == 1
    assert db_path.exists()
    assert db.get_statistics()["total_count"] == 1

    FeedbackDatabase.reset_singleton()


def test_feedback_database_reset_handles_uninitialized_instance():
    FeedbackDatabase.reset_singleton()
    instance = object.__new__(FeedbackDatabase)
    FeedbackDatabase._instance = instance

    FeedbackDatabase.reset_singleton()

    assert FeedbackDatabase._instance is None
