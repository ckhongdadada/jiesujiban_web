from __future__ import annotations

from src.jsjb.feedback.repository import FeedbackDatabase


def test_feedback_dashboard_aggregates_reply_and_rag_signals(tmp_path):
    FeedbackDatabase.reset_singleton()
    db_path = tmp_path / "feedback_dashboard.db"
    db = FeedbackDatabase(str(db_path))

    feedback_id = db.add_feedback(
        {
            "tag": "投诉",
            "title": "垃圾清运不及时",
            "body": "朝阳区某小区垃圾清运不及时",
            "reply": "已安排处理",
            "unit": "环卫中心",
            "district": "朝阳区",
            "is_helpful": False,
            "feedback_type": "reply_quality",
            "comments": "回复缺少具体措施",
        }
    )
    db.save_reply_error_analysis(
        feedback_id,
        {
            "summary": "存在事实缺失",
            "severity": "medium",
            "error_types": ["missing_action", "unit_mismatch"],
            "errors": [],
        },
        reference_reply="已明确责任单位和处理时限",
    )
    db.record_doc_feedback("doc-good", True, query="垃圾清运")
    db.record_doc_feedback("doc-good", True, query="垃圾清运")
    db.record_doc_feedback("doc-bad", False, query="垃圾清运")
    db.queue_knowledge_graph_facts(
        feedback_id=feedback_id,
        analysis_id=1,
        analysis={
            "reference_facts": [
                {
                    "fact_type": "responsible_unit",
                    "content": {"entity_name": "垃圾清运", "fact_value": "环卫中心"},
                    "context_excerpt": "由环卫中心负责垃圾清运",
                }
            ]
        },
    )

    dashboard = db.get_feedback_dashboard(limit=10)

    assert dashboard["status"] == "ok"
    assert dashboard["feedback_statistics"]["total_count"] == 1
    assert dashboard["reply_error_analysis_count"] == 1
    assert dashboard["pending_kg_fact_count"] == 1
    assert dashboard["doc_feedback_count"] == 3
    assert dashboard["quality_error_distribution"][0]["error_type"] in {"missing_action", "unit_mismatch"}

    doc_ranking = {item["doc_id"]: item for item in dashboard["doc_feedback_ranking"]}
    assert doc_ranking["doc-good"]["net_votes"] == 2
    assert doc_ranking["doc-good"]["feedback_boost"] == 0.1
    assert doc_ranking["doc-bad"]["feedback_boost"] == -0.05

    db.close()
    FeedbackDatabase.reset_singleton()
