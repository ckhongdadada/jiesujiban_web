from __future__ import annotations

from src.jsjb.feedback.attribution import FeedbackAttributor
from src.jsjb.feedback.automation import FeedbackAutomationRouter
from src.jsjb.feedback.repository import FeedbackDatabase, get_feedback_database
from src.jsjb.reply_generation.error_analysis import ReplyErrorExtractor


def test_feedback_attributor_extracts_corrected_unit_and_routes():
    attribution = FeedbackAttributor().analyze(
        feedback_record={
            "unit": "市医保局",
            "district": "大兴区",
            "is_helpful": False,
            "feedback_type": "单位错误",
            "comments": "这个问题正确单位是大兴区政府，不应该给市医保局。",
        },
        request_data={
            "units": [
                {"unit": "市医保局", "confidence": 0.3},
                {"unit": "市民政局", "confidence": 0.2},
                {"unit": "卫生健康委", "confidence": 0.1},
            ]
        },
        quality_attribution={},
    )

    assert attribution["corrected_unit"] == "大兴区政府"
    assert attribution["primary_error"] == "unit_top3_missing"
    assert "unit_wrong" in attribution["error_tags"]
    assert "add_classifier_training_candidate" in attribution["recommended_actions"]


def test_feedback_router_queues_classifier_reply_and_knowledge_candidates(tmp_path):
    FeedbackDatabase.reset_singleton()
    db = get_feedback_database(str(tmp_path / "feedback.db"))
    record = {
        "timestamp": "2026-05-06T12:00:00",
        "tag": "物业管理",
        "title": "小区物业问题",
        "body": "大兴区某小区物业长期不维修。",
        "reply": "已转相关部门。",
        "unit": "市医保局",
        "district": "大兴区",
        "is_helpful": False,
        "feedback_type": "单位错误",
        "comments": "正确单位是大兴区政府，回复也太空泛。",
    }
    feedback_id = db.add_feedback(record)

    router = FeedbackAutomationRouter(
        feedback_db=db,
        reply_error_extractor=ReplyErrorExtractor(),
    )
    result = router.run(
        feedback_id=feedback_id,
        feedback_record=record,
        request_data={
            "units": [{"unit": "市医保局", "confidence": 0.31}],
            "retrieval": [{"id": "doc-1", "title": "物业案例", "score": 0.5}],
            "location": {"district": "大兴区"},
        },
    )

    assert result.feedback_attribution_id is not None
    assert result.classifier_candidate_id is not None
    assert result.reply_error_case_id is not None
    assert result.queued_fact_count >= 1
    dashboard = db.get_feedback_dashboard()
    assert dashboard["feedback_attribution_count"] == 1
    assert dashboard["pending_classifier_candidate_count"] == 1
    assert dashboard["pending_reply_error_case_count"] == 1
    assert dashboard["pending_kg_fact_count"] >= 1

    db.close()
    FeedbackDatabase.reset_singleton()
