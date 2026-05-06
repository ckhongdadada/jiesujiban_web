from __future__ import annotations

import json

from src.jsjb.evaluation.reply_quality import (
    ReplyQualityEvaluator,
    compute_bleu,
    compute_rouge,
    load_reply_quality_config_file,
    read_quality_rows,
    write_quality_outputs,
)


def test_compute_rouge_and_bleu_for_similar_reply():
    generated = "经现场核实，已安排物业维修电梯，并督促后续加强巡查。"
    reference = "经现场核实，已安排物业维修电梯，后续将加强巡查。"

    rouge = compute_rouge(generated, reference)
    bleu = compute_bleu(generated, reference)

    assert rouge["rouge_1"] > 0.7
    assert rouge["rouge_l"] > 0.7
    assert bleu > 0.4


def test_reply_quality_evaluator_builds_standard_case():
    evaluator = ReplyQualityEvaluator()
    result = evaluator.evaluate_rows(
        [
            {
                "id": "case-1",
                "tag": "物业管理",
                "title": "电梯故障",
                "body": "小区电梯多次故障。",
                "unit": "住建委",
                "district": "大兴区",
                "generated_reply": "经现场核实，已督促物业维修电梯，并加强后续巡查。",
                "reference_reply": "经现场核实，已安排物业维修电梯，后续将加强巡查。",
                "rag_docs": json.dumps([{"title": "物业维修案例", "content": "物业维修 电梯 巡查"}], ensure_ascii=False),
            }
        ]
    )

    assert result["metrics"]["sample_count"] == 1
    assert result["cases"][0]["metrics"]["quality_score"] > 0
    assert "rouge_l" in result["cases"][0]["metrics"]
    assert "recommendation" in result["metrics"]


def test_write_quality_outputs(tmp_path):
    evaluator = ReplyQualityEvaluator()
    result = evaluator.evaluate_rows(
        [
            {
                "generated_reply": "已协调物业进行维修。",
                "reference_reply": "已协调物业进行维修。",
            }
        ]
    )

    written = write_quality_outputs(result, str(tmp_path / "quality"))

    assert (tmp_path / "quality" / "metrics.json").exists()
    assert (tmp_path / "quality" / "cases.jsonl").exists()
    assert (tmp_path / "quality" / "report.md").exists()
    assert "metrics_json" in written


def test_read_quality_rows_accepts_utf8_bom_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"generated_reply":"已处理","reference_reply":"已处理"}\n', encoding="utf-8-sig")

    rows = read_quality_rows(str(path))

    assert rows[0]["generated_reply"] == "已处理"


def test_load_reply_quality_config_file(tmp_path):
    path = tmp_path / "reply_quality.json"
    path.write_text(
        json.dumps(
            {
                "min_length": 10,
                "max_length": 120,
                "use_bge_similarity": False,
                "quality_score_weights": {"semantic_similarity": 1.0},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    config = load_reply_quality_config_file(str(path))

    assert config.min_length == 10
    assert config.max_length == 120
    assert config.weights == {"semantic_similarity": 1.0}
