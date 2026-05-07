from __future__ import annotations

from src.jsjb.reply_generation.atomic_facts import split_reply_to_atomic_facts
from src.jsjb.reply_generation import qwen_lora


def test_split_reply_to_atomic_facts_extracts_status_action_unit_and_time():
    reply = "经朝阳区城管委核实，已安排物业清理垃圾，预计5月10日完成整改。"

    facts = split_reply_to_atomic_facts(reply, {"evidence_items": [{"evidence_id": "E1"}]})
    fact_types = {fact["fact_type"] for fact in facts}

    assert {"district", "unit", "action", "time"} <= fact_types
    assert any(fact["value"] == "朝阳区" for fact in facts)
    assert any("已安排" in fact["value"] for fact in facts)
    assert all(fact["fact_id"].startswith("A") for fact in facts)


def test_generate_reply_with_context_returns_atomic_facts_in_fallback(monkeypatch):
    monkeypatch.setattr(qwen_lora, "load_generator", lambda *args, **kwargs: False)

    result = qwen_lora.generate_reply_with_context(
        tag="投诉",
        title="垃圾清运不及时",
        body="小区垃圾堆放时间较长。",
        unit="城管委",
        location_result={"district": "大兴区"},
        retrieval_hits=[{"doc_id": "doc-1", "title": "垃圾清运案例", "district": "大兴区", "snippet": "已督促物业加强垃圾清运。"}],
        return_dict=True,
    )

    assert result["atomic_facts"]
    assert result["atomic_facts"][0]["fact_id"].startswith("A")
