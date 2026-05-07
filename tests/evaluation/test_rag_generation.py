
from __future__ import annotations

import json

from src.jsjb.evaluation.rag_generation import (
    Phase1RagGenerationEvaluator,
    evidence_coverage,
    read_phase1_cases,
    write_phase1_outputs,
)


def test_phase1_evaluator_scores_retrieval_and_reply():
    record = {
        "id": "case-a",
        "title": "\u5783\u573e\u6e05\u8fd0\u95ee\u9898",
        "expected_district": "\u5927\u5174\u533a",
        "expected_unit": "\u57ce\u7ba1\u59d4",
        "expected_retrieval_keywords": ["\u5783\u573e", "\u6e05\u8fd0"],
        "expected_evidence_terms": ["\u5783\u573e", "\u6e05\u8fd0"],
        "response": {
            "location": {"district": "\u5927\u5174\u533a"},
            "units": [{"unit": "\u57ce\u7ba1\u59d4"}, {"unit": "\u8857\u9053\u529e"}],
            "reply": "\u7ecf\u6838\u5b9e\uff0c\u5df2\u534f\u8c03\u73af\u536b\u5355\u4f4d\u52a0\u5f3a\u5783\u573e\u6e05\u8fd0\u548c\u6876\u7ad9\u4fdd\u6d01\u3002",
            "retrieval": [
                {
                    "doc_id": "doc-1",
                    "title": "\u5783\u573e\u6e05\u8fd0\u6848\u4f8b",
                    "district": "\u5927\u5174\u533a",
                    "content": "\u5783\u573e \u6e05\u8fd0 \u6876\u7ad9 \u4fdd\u6d01",
                    "score": 0.8,
                    "dense_score": 0.7,
                    "sparse_score": 0.6,
                }
            ],
        },
    }

    result = Phase1RagGenerationEvaluator().evaluate_records([record])
    case = result["cases"][0]

    assert result["summary"]["sample_count"] == 1
    assert case["metrics"]["retrieval_precision_at_k"] == 1.0
    assert case["metrics"]["retrieval_top1_same_district"] == 1.0
    assert case["metrics"]["unit_top1_hit"] == 1.0
    assert case["metrics"]["evidence_coverage"] > 0.5
    assert case["retrieval_observability"][0]["score_breakdown"]["dense_score"] == 0.7


def test_evidence_coverage_requires_grounding():
    hits = [{"content": "\u7535\u68af \u7ef4\u4fee \u7269\u4e1a"}]
    assert evidence_coverage("\u5df2\u5b89\u6392\u7535\u68af\u7ef4\u4fee", hits, ["\u7535\u68af", "\u7ef4\u4fee"]) == 1.0
    assert evidence_coverage("\u5df2\u5b89\u6392\u5783\u573e\u6e05\u8fd0", hits, ["\u5783\u573e", "\u6e05\u8fd0"]) == 0.0


def test_phase1_read_and_write_outputs(tmp_path):
    input_path = tmp_path / "cases.jsonl"
    input_path.write_text(
        json.dumps({"id": "x", "response": {"reply": "\u5df2\u5904\u7406", "retrieval": []}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = read_phase1_cases(input_path)
    result = Phase1RagGenerationEvaluator().evaluate_records(rows)
    written = write_phase1_outputs(result, tmp_path / "out")

    assert len(rows) == 1
    assert (tmp_path / "out" / "eval_latest.json").exists()
    assert (tmp_path / "out" / "eval_cases_latest.jsonl").exists()
    assert (tmp_path / "out" / "eval_report_latest.md").exists()
    assert "latest_json" in written
