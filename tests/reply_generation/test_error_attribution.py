from __future__ import annotations

from src.jsjb.reply_generation.error_analysis import ReplyErrorExtractor


def test_error_attribution_without_reference_uses_grounding_and_actionability():
    extractor = ReplyErrorExtractor()

    result = extractor.analyze(
        generated_reply="已转相关部门研究，请耐心等待。",
        reference_reply="",
        retrieval_hits=[],
        location_result={"district": "朝阳区"},
        expected_unit="城管委",
        feedback_type="回复无用",
        comments="缺少事实依据",
    )

    assert result["needs_review"] is True
    assert result["severity"] in {"medium", "high"}
    assert "weak_evidence" in result["error_types"]
    assert "low_actionability" in result["error_types"]
    assert result["quality_dimensions"]["grounding"] == "weak"
    assert "manual_review" in result["routing_recommendations"]


def test_error_attribution_detects_location_mismatch():
    extractor = ReplyErrorExtractor()

    result = extractor.analyze(
        generated_reply="经海淀区相关部门核实，已安排清理。",
        reference_reply="经朝阳区相关部门核实，已安排清理。",
        retrieval_hits=[{"id": "doc-1", "snippet": "朝阳区该小区已安排垃圾清运。", "score": 0.9}],
        location_result={"district": "朝阳区"},
    )

    assert "district_or_street_mismatch" in result["error_types"]
    assert result["quality_dimensions"]["location_consistency"] == "fail"
