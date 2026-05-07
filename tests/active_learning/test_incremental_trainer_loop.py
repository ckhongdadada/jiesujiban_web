from __future__ import annotations

import pandas as pd

from src.jsjb.active_learning.sample_collector import SampleCollector
from src.jsjb.active_learning.trainer import IncrementalTrainer


def test_incremental_trainer_merges_badge_annotations_into_classifier_csv(tmp_path):
    base_path = tmp_path / "base.csv"
    pd.DataFrame(
        [
            {
                "留言标签": "环境卫生",
                "留言标题": "垃圾桶满溢",
                "留言正文": "小区垃圾桶满溢无人清运",
                "官方回复单位": "城管委",
            }
        ]
    ).to_csv(base_path, index=False, encoding="utf-8-sig")

    collector = SampleCollector(str(tmp_path / "active_learning.db"))
    collector.add_sample(
        sample_id="badge-1",
        tag="物业管理",
        title="电梯故障",
        body="电梯经常停运",
        district="海淀区",
        predicted_unit="街道办",
        confidence=0.31,
        prediction_probs={"街道办": 0.31, "住建委": 0.29},
        sample_source="classifier_badge",
        trigger_reason="badge_uncertainty_diversity",
    )
    assert collector.annotate_sample("badge-1", correct_unit="住建委", annotated_by="tester")

    trainer = IncrementalTrainer(
        sample_collector=collector,
        base_data_path=str(base_path),
        model_dir=str(tmp_path / "model"),
        training_script="training/train_unit_classifier_hybrid.py",
    )
    output_path = trainer.prepare_training_data(str(tmp_path / "merged.csv"))
    merged = pd.read_csv(output_path)

    assert len(merged) == 2
    assert set(merged["官方回复单位"]) == {"城管委", "住建委"}
    assert "active_learning_sample_id" in merged.columns
    assert "badge-1" in set(merged["active_learning_sample_id"].fillna(""))
