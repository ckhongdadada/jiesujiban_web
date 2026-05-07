from __future__ import annotations

from src.jsjb.reply_generation.evidence_grounding import (
    bind_reply_to_evidence,
    build_evidence_plan,
    format_evidence_plan_for_prompt,
)
from src.jsjb.reply_generation import qwen_lora


def test_build_evidence_plan_extracts_atomic_facts():
    hits = [
        {
            "doc_id": "case-1",
            "title": "道路建设进展",
            "doc_type": "案例",
            "district": "朝阳区",
            "unit": "朝阳区城管委",
            "score": 0.91,
            "snippet": "朝阳路道路工程已完工并通车。",
            "project_status": "已完成",
        }
    ]

    plan = build_evidence_plan(hits, unit="朝阳区城管委", location_result={"district": "朝阳区"})

    assert plan["coverage"] == "structured"
    assert plan["evidence_items"][0]["evidence_id"] == "E1"
    facts = plan["evidence_items"][0]["atomic_facts"]
    assert any(fact["fact_type"] == "project_status" and fact["value"] == "已完成" for fact in facts)
    assert any("朝阳区" in claim for claim in plan["allowed_claims"])


def test_format_evidence_plan_includes_constraints_not_empty():
    plan = build_evidence_plan(
        [{"title": "社区资源情况", "district": "西城区", "snippet": "社区图书馆已开放运营。", "resource_status": "已运营"}],
        unit="文旅局",
        location_result={"district": "西城区"},
    )

    rendered = format_evidence_plan_for_prompt(plan)

    assert "证据计划" in rendered
    assert "事实约束" in rendered
    assert "resource_status" in rendered or "资源" in rendered


def test_bind_reply_to_evidence_flags_unsupported_factual_sentence():
    plan = build_evidence_plan(
        [{"title": "垃圾清运案例", "district": "丰台区", "snippet": "已安排物业加强垃圾清运。"}],
        unit="丰台区城管委",
        location_result={"district": "丰台区"},
    )
    reply = "已安排物业加强垃圾清运。该道路将于2026年6月30日通车。"

    report = bind_reply_to_evidence(reply, plan)

    assert report["needs_review"] is True
    assert report["unsupported_sentences"]
    assert any("2026年6月30日" in item["sentence"] for item in report["unsupported_sentences"])


def test_generate_reply_fallback_returns_evidence_verification(monkeypatch):
    monkeypatch.setattr(qwen_lora, "load_generator", lambda *args, **kwargs: False)
    result = qwen_lora.generate_reply_with_context(
        tag="投诉",
        title="垃圾清运不及时",
        body="小区垃圾堆放时间较长。",
        unit="城管委",
        location_result={"district": "大兴区"},
        retrieval_hits=[
            {
                "doc_id": "doc-1",
                "title": "垃圾清运案例",
                "district": "大兴区",
                "snippet": "大兴区已督促物业加强垃圾清运。",
            }
        ],
        return_dict=True,
    )

    assert result["fallback"] is True
    assert result["evidence_plan"]["evidence_items"]
    assert "evidence_bindings" in result["verification"]
    assert "evidence_coverage_rate" in result["verification"]
