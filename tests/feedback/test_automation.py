from __future__ import annotations

from src.jsjb.active_learning.sample_collector import SampleCollector
from src.jsjb.feedback.automation import FeedbackAutomationRouter
from src.jsjb.feedback.repository import FeedbackDatabase, get_feedback_database
from src.jsjb.reply_generation.error_analysis import ReplyErrorExtractor


def _feedback_record(**overrides):
    record = {
        "timestamp": "2026-05-06T12:00:00",
        "tag": "环境卫生",
        "title": "小区垃圾清运不及时",
        "body": "朝阳区某小区垃圾桶长期满溢，希望尽快处理。",
        "reply": "正在推进处理，请耐心等待。",
        "unit": "城管委",
        "district": "朝阳区",
        "is_helpful": False,
        "feedback_type": "回复无用",
        "comments": "回复没有具体办理动作，参考材料也不太相关。",
        "client_ip": "127.0.0.1",
        "user_agent": "pytest",
        "processing_time": 1.0,
    }
    record.update(overrides)
    return record


def test_feedback_router_records_doc_feedback_and_active_learning(tmp_path):
    FeedbackDatabase.reset_singleton()
    db = get_feedback_database(str(tmp_path / "feedback.db"))
    collector = SampleCollector(str(tmp_path / "active_learning.db"))
    record = _feedback_record()
    feedback_id = db.add_feedback(record)

    router = FeedbackAutomationRouter(
        feedback_db=db,
        reply_error_extractor=ReplyErrorExtractor(),
        sample_collector=collector,
    )
    result = router.run(
        feedback_id=feedback_id,
        feedback_record=record,
        request_data={
            "retrieval": [
                {"id": "doc-1", "title": "垃圾清运案例", "score": 0.82, "district": "朝阳区"},
                {"id": "doc-2", "title": "环境卫生政策", "score": 0.61, "district": "朝阳区"},
            ],
            "units": [{"unit": "城管委", "confidence": 0.42}],
            "location": {"district": "朝阳区"},
        },
    )

    assert result.doc_feedback_recorded == 2
    assert result.active_learning_collected is True
    assert result.reply_error_analysis_id is not None
    assert "reply_quality_attribution_saved" in result.routing_reasons
    assert db.get_doc_feedback_scores()["doc-1"] < 0
    assert collector.get_statistics()["pending_samples"] == 1

    db.close()
    FeedbackDatabase.reset_singleton()


def test_feedback_router_saves_quality_attribution_without_reference_reply(tmp_path):
    FeedbackDatabase.reset_singleton()
    db = get_feedback_database(str(tmp_path / "feedback.db"))
    record = _feedback_record(reply="已转相关部门研究。", comments="没有任何具体事实依据")
    feedback_id = db.add_feedback(record)

    router = FeedbackAutomationRouter(
        feedback_db=db,
        reply_error_extractor=ReplyErrorExtractor(),
        sample_collector=None,
    )
    result = router.run(
        feedback_id=feedback_id,
        feedback_record=record,
        request_data={"retrieval": [], "location": {"district": "朝阳区"}},
    )

    assert result.reply_error_analysis_id is not None
    assert result.quality_attribution is not None
    assert "weak_evidence" in result.quality_attribution["error_types"]
    assert "low_actionability" in result.quality_attribution["error_types"]
    assert result.quality_attribution["quality_dimensions"]["grounding"] == "weak"

    saved = db.get_reply_error_analysis(feedback_id)
    assert saved is not None
    assert "weak_evidence" in saved["error_types"]
    assert saved["quality_dimensions"]["grounding"] == "weak"
    assert "manual_review" in saved["routing_recommendations"]

    db.close()
    FeedbackDatabase.reset_singleton()
