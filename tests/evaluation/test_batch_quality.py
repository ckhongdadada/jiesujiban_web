from __future__ import annotations

from src.jsjb.evaluation.batch_quality import build_batch_summary, evaluate_response_quality, flatten_api_response


def test_evaluate_response_quality_adds_attribution():
    response = {
        "status": "ok",
        "location": {"district": "朝阳区"},
        "units": [{"unit": "城管委", "confidence": 0.55}],
        "retrieval": [],
        "reply": "已转相关部门研究，请耐心等待。",
    }

    quality = evaluate_response_quality(response, expected_unit="城管委")

    assert quality["needs_review"] is True
    assert "weak_evidence" in quality["error_types"]
    assert "low_actionability" in quality["error_types"]


def test_flatten_and_summary_counts_quality_errors():
    response = {
        "status": "ok",
        "location": {"district": "朝阳区", "confidence": 0.9},
        "units": [{"unit": "城管委", "confidence": 0.55}, {"unit": "街道办", "confidence": 0.2}],
        "retrieval": [{"title": "案例", "score": 0.8}],
        "evidence_strength": "strong",
        "reply": "已安排清理。",
        "processing_time": {"total": 1.2},
    }
    flat = flatten_api_response(response)
    assert flat["top1_unit"] == "城管委"
    assert flat["top3_units"] == ["城管委", "街道办"]

    summary = build_batch_summary(
        [
            {
                "status": "ok",
                "elapsed_seconds": 1.0,
                "needs_review": True,
                "quality_attribution": {"error_types": ["weak_evidence"]},
            },
            {"status": "error", "elapsed_seconds": 0.1},
        ],
        elapsed_seconds=1.1,
    )
    assert summary["sample_count"] == 2
    assert summary["ok_count"] == 1
    assert summary["quality_attribution_distribution"]["weak_evidence"] == 1
