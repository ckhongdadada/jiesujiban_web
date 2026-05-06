from __future__ import annotations

import json

from src.jsjb.active_learning.automation import ActiveLearningAutomationService
from src.jsjb.active_learning.sample_collector import SampleCollector


def test_review_plan_clusters_pending_samples(tmp_path):
    collector = SampleCollector(str(tmp_path / "active_learning.db"))
    collector.add_sample(
        sample_id="s1",
        tag="环境卫生",
        title="垃圾清运",
        body="垃圾桶满溢",
        district="朝阳区",
        predicted_unit="城管委",
        confidence=0.32,
        prediction_probs={"城管委": 0.32, "街道办": 0.28},
        sample_source="generated_reply_quality_issue",
        trigger_reason="low_actionability",
        risk_flags={"reply_error_types": ["low_actionability"], "reply_error_severity": "medium"},
    )
    collector.add_sample(
        sample_id="s2",
        tag="物业管理",
        title="电梯故障",
        body="电梯经常停运",
        district="海淀区",
        predicted_unit="住建委",
        confidence=0.21,
        prediction_probs={"住建委": 0.21, "街道办": 0.19},
        sample_source="classifier_low_confidence",
        trigger_reason="confidence below threshold",
        risk_flags={"feedback_type": "low_confidence"},
    )

    plan = ActiveLearningAutomationService(collector).build_review_plan(limit=10)

    assert plan["status"] == "ok"
    assert plan["sample_count"] == 2
    assert plan["summary"]["cluster_count"] == 2
    assert plan["next_action"]["action"] in {"annotate_pending", "annotate_negative_feedback"}
    assert {sample["sample_id"] for sample in plan["samples"]} == {"s1", "s2"}


def test_prepare_training_package_exports_annotated_samples(tmp_path):
    collector = SampleCollector(str(tmp_path / "active_learning.db"))
    collector.add_sample(
        sample_id="s1",
        tag="环境卫生",
        title="垃圾清运",
        body="垃圾桶满溢",
        district="朝阳区",
        predicted_unit="街道办",
        confidence=0.32,
    )
    assert collector.annotate_sample("s1", correct_unit="城管委", annotated_by="tester", notes="人工确认")

    service = ActiveLearningAutomationService(collector)
    result = service.prepare_training_package(output_dir=str(tmp_path / "exports"), min_samples=1)

    assert result.status == "ok"
    assert result.sample_count == 1
    records = [json.loads(line) for line in open(result.data_path, encoding="utf-8") if line.strip()]
    assert records[0]["unit"] == "城管委"
    manifest = json.loads(open(result.manifest_path, encoding="utf-8").read())
    assert manifest["sample_count"] == 1
