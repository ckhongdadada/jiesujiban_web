
"""Phase-1 RAG and generation evaluation utilities.

This module is intentionally independent from Flask and model loading. It can
score saved /api/analyze responses, offline generated replies, or responses
collected by the CLI runner. The goal is a repeatable evaluation baseline, not
an ad-hoc smoke test.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.jsjb.evaluation.reply_quality import format_compliance_score

BEIJING_DISTRICTS = [
    "\u4e1c\u57ce\u533a", "\u897f\u57ce\u533a", "\u671d\u9633\u533a", "\u4e30\u53f0\u533a",
    "\u77f3\u666f\u5c71\u533a", "\u6d77\u6dc0\u533a", "\u95e8\u5934\u6c9f\u533a", "\u623f\u5c71\u533a",
    "\u901a\u5dde\u533a", "\u987a\u4e49\u533a", "\u660c\u5e73\u533a", "\u5927\u5174\u533a",
    "\u6000\u67d4\u533a", "\u5e73\u8c37\u533a", "\u5bc6\u4e91\u533a", "\u5ef6\u5e86\u533a",
]

STOP_TERMS = {
    "\u95ee\u9898", "\u60c5\u51b5", "\u8fdb\u884c", "\u5df2\u7ecf", "\u76f8\u5173", "\u90e8\u95e8",
    "\u5904\u7406", "\u53cd\u6620", "\u8bc9\u6c42", "\u56de\u590d", "\u6838\u5b9e", "\u529e\u7406",
    "\u8fdb\u4e00\u6b65", "\u73b0\u573a", "\u7fa4\u4f17", "\u5e02\u6c11", "\u5de5\u4f5c", "\u52a0\u5f3a",
}


@dataclass
class Phase1EvaluationConfig:
    retrieval_top_k: int = 3
    min_evidence_coverage: float = 0.35
    min_retrieval_precision: float = 0.34
    min_format_score: float = 0.85
    metrics: list[str] = field(
        default_factory=lambda: [
            "retrieval_precision_at_k",
            "retrieval_top1_same_district",
            "retrieval_topk_any_same_district",
            "evidence_coverage",
            "format_compliance",
            "unit_top1_hit",
            "unit_top3_hit",
            "other_district_leak",
        ]
    )


class Phase1RagGenerationEvaluator:
    """Evaluate retrieval evidence and generated reply observability metrics."""

    def __init__(self, config: Phase1EvaluationConfig | None = None) -> None:
        self.config = config or Phase1EvaluationConfig()

    def evaluate_records(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        cases = [self.evaluate_case(record, index=index) for index, record in enumerate(records)]
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "evaluation_name": "phase1_rag_generation",
            "config": self.config.__dict__,
            "summary": self.build_summary(cases),
            "cases": cases,
        }

    def evaluate_case(self, record: dict[str, Any], index: int = 0) -> dict[str, Any]:
        response = record.get("response") if isinstance(record.get("response"), dict) else record
        retrieval = parse_retrieval(response.get("retrieval") or record.get("retrieval") or [])
        reply = str(response.get("reply") or record.get("generated_reply") or "")
        expected_district = pick(record, "expected_district", "district")
        expected_unit = pick(record, "expected_unit", "unit")
        expected_doc_ids = normalize_list(record.get("expected_doc_ids") or record.get("expected_doc_id") or [])
        expected_keywords = normalize_list(record.get("expected_retrieval_keywords") or [])
        expected_evidence_terms = normalize_list(record.get("expected_evidence_terms") or expected_keywords)
        location = response.get("location") if isinstance(response.get("location"), dict) else {}
        units = response.get("units") if isinstance(response.get("units"), list) else []

        top_k_hits = retrieval[: self.config.retrieval_top_k]
        retrieval_precision = retrieval_precision_at_k(top_k_hits, expected_doc_ids, expected_keywords)
        top1_same = same_district(top_k_hits[0], expected_district) if top_k_hits else False
        topk_same = any(same_district(hit, expected_district) for hit in top_k_hits) if top_k_hits else False
        coverage = evidence_coverage(reply, top_k_hits, expected_evidence_terms)
        format_score = format_compliance_score(reply)
        unit_top1 = unit_hit(units[:1], expected_unit)
        unit_top3 = unit_hit(units[:3], expected_unit)
        other_districts = other_district_mentions(reply, expected_district)

        metrics = {
            "retrieval_precision_at_k": round(retrieval_precision, 4),
            "retrieval_top1_same_district": float(top1_same),
            "retrieval_topk_any_same_district": float(topk_same),
            "evidence_coverage": round(coverage, 4),
            "format_compliance": round(format_score, 4),
            "unit_top1_hit": float(unit_top1),
            "unit_top3_hit": float(unit_top3),
            "other_district_leak": float(bool(other_districts)),
        }
        risk_flags = build_risk_flags(metrics, self.config)
        score_breakdown = [extract_hit_observability(hit, rank=i + 1) for i, hit in enumerate(top_k_hits)]

        return {
            "index": index,
            "id": record.get("id") or response.get("trace_id") or index,
            "title": pick(record, "title", "\u7559\u8a00\u6807\u9898"),
            "body": pick(record, "body", "\u7559\u8a00\u6b63\u6587"),
            "tag": pick(record, "tag", "\u7559\u8a00\u6807\u7b7e"),
            "expected_district": expected_district,
            "detected_district": location.get("district", ""),
            "expected_unit": expected_unit,
            "top3_units": [unit.get("unit", "") for unit in units[:3] if isinstance(unit, dict)],
            "reply": reply,
            "metrics": metrics,
            "risk_flags": risk_flags,
            "needs_review": bool(risk_flags),
            "other_district_mentions": other_districts,
            "retrieval_observability": score_breakdown,
            "retrieval_count": len(retrieval),
            "reply_mode": response.get("reply_mode", ""),
            "evidence_strength": response.get("evidence_strength", ""),
            "processing_time": response.get("processing_time", {}),
        }

    @staticmethod
    def build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
        metric_keys = [
            "retrieval_precision_at_k",
            "retrieval_top1_same_district",
            "retrieval_topk_any_same_district",
            "evidence_coverage",
            "format_compliance",
            "unit_top1_hit",
            "unit_top3_hit",
            "other_district_leak",
        ]
        averages = {}
        for key in metric_keys:
            values = [float(case.get("metrics", {}).get(key, 0.0) or 0.0) for case in cases]
            averages[key] = round(sum(values) / len(values), 4) if values else 0.0

        risk_counts: Counter[str] = Counter()
        for case in cases:
            for flag in case.get("risk_flags", []) or []:
                risk_counts[flag] += 1

        worst_cases = sorted(
            [
                {
                    "id": case.get("id"),
                    "title": case.get("title"),
                    "risk_flags": case.get("risk_flags", []),
                    "evidence_coverage": case.get("metrics", {}).get("evidence_coverage", 0.0),
                    "retrieval_precision_at_k": case.get("metrics", {}).get("retrieval_precision_at_k", 0.0),
                    "format_compliance": case.get("metrics", {}).get("format_compliance", 0.0),
                }
                for case in cases
                if case.get("risk_flags")
            ],
            key=lambda item: (item["evidence_coverage"], item["retrieval_precision_at_k"], item["format_compliance"]),
        )[:20]

        return {
            "sample_count": len(cases),
            "metric_averages": averages,
            "risk_distribution": dict(risk_counts.most_common()),
            "needs_review_count": sum(1 for case in cases if case.get("needs_review")),
            "needs_review_rate": round(sum(1 for case in cases if case.get("needs_review")) / max(len(cases), 1), 4),
            "worst_cases": worst_cases,
        }


def pick(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            return [item.strip() for item in re.split(r"[,?;?|]", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def parse_retrieval(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            return []
    return []


def retrieval_precision_at_k(hits: list[dict[str, Any]], expected_doc_ids: list[str], expected_keywords: list[str]) -> float:
    if not hits:
        return 0.0
    if not expected_doc_ids and not expected_keywords:
        return 1.0 if hits else 0.0
    matched = 0
    expected_id_set = set(expected_doc_ids)
    for hit in hits:
        doc_id = str(hit.get("doc_id") or hit.get("id") or "")
        searchable = " ".join(
            str(hit.get(key) or "") for key in ["title", "snippet", "content", "full_content", "source", "doc_type"]
        )
        if doc_id in expected_id_set or any(keyword and keyword in searchable for keyword in expected_keywords):
            matched += 1
    return matched / max(len(hits), 1)


def same_district(hit: dict[str, Any], expected_district: str) -> bool:
    if not expected_district:
        return False
    district = str(hit.get("district") or "")
    return district == expected_district or district in {"\u5317\u4eac\u5e02", "\u5168\u5e02"}


def unit_hit(units: list[dict[str, Any]], expected_unit: str) -> bool:
    if not expected_unit:
        return False
    for unit in units:
        name = str(unit.get("unit") or unit.get("name") or "")
        if name and (name == expected_unit or name in expected_unit or expected_unit in name):
            return True
    return False


def evidence_coverage(reply: str, hits: list[dict[str, Any]], expected_terms: list[str] | None = None) -> float:
    if not reply:
        return 0.0
    evidence_text = "\n".join(
        str(hit.get("full_content") or hit.get("content") or hit.get("parent_context") or hit.get("snippet") or hit.get("title") or "")
        for hit in hits
    )
    if not evidence_text.strip():
        return 0.0
    if expected_terms:
        terms = [term for term in expected_terms if term]
    else:
        terms = extract_salient_terms(reply)
    if not terms:
        return 0.5
    supported = sum(1 for term in terms if term in evidence_text)
    return supported / max(len(terms), 1)


def extract_salient_terms(text: str) -> list[str]:
    terms = set()
    for token in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,12}", text or ""):
        if token not in STOP_TERMS and len(token) >= 2:
            terms.add(token)
    return sorted(terms)[:20]


def other_district_mentions(reply: str, expected_district: str) -> list[str]:
    return [district for district in BEIJING_DISTRICTS if district != expected_district and district in (reply or "")]


def build_risk_flags(metrics: dict[str, float], config: Phase1EvaluationConfig) -> list[str]:
    flags = []
    if metrics.get("retrieval_precision_at_k", 0.0) < config.min_retrieval_precision:
        flags.append("low_retrieval_precision")
    if metrics.get("retrieval_topk_any_same_district", 0.0) < 1.0:
        flags.append("no_same_district_evidence")
    if metrics.get("evidence_coverage", 0.0) < config.min_evidence_coverage:
        flags.append("low_evidence_coverage")
    if metrics.get("format_compliance", 0.0) < config.min_format_score:
        flags.append("format_violation")
    if metrics.get("other_district_leak", 0.0) > 0:
        flags.append("other_district_leak")
    return flags


def extract_hit_observability(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    score_breakdown = hit.get("score_breakdown") if isinstance(hit.get("score_breakdown"), dict) else {}
    if not score_breakdown:
        score_breakdown = {
            "final_score": hit.get("score", 0.0),
            "dense_score": hit.get("dense_score", 0.0),
            "sparse_score": hit.get("sparse_score", 0.0),
            "rerank_score": hit.get("rerank_score", 0.0),
            "feedback_boost": hit.get("feedback_boost", 0.0),
        }
    return {
        "rank": rank,
        "doc_id": hit.get("doc_id") or hit.get("id") or "",
        "title": hit.get("title", ""),
        "doc_type": hit.get("doc_type", ""),
        "district": hit.get("district", ""),
        "source": hit.get("source", ""),
        "retrieval_backend": hit.get("retrieval_backend", ""),
        "retrieval_granularity": hit.get("retrieval_granularity", ""),
        "child_chunk_id": hit.get("child_chunk_id", ""),
        "matched_terms": hit.get("matched_terms", []),
        "score_breakdown": score_breakdown,
        "snippet": hit.get("child_snippet") or hit.get("snippet", ""),
        "parent_context": hit.get("parent_context", ""),
    }


def read_phase1_cases(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8-sig") as fp:
            for line in fp:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else payload.get("cases", [])


def write_phase1_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "eval_latest.json"
    archive_path = output / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    cases_path = output / "eval_cases_latest.jsonl"
    report_path = output / "eval_report_latest.md"

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    latest_path.write_text(payload, encoding="utf-8")
    archive_path.write_text(payload, encoding="utf-8")
    with cases_path.open("w", encoding="utf-8") as fp:
        for case in result.get("cases", []):
            fp.write(json.dumps(case, ensure_ascii=False) + "\n")
    report_path.write_text(render_phase1_markdown(result), encoding="utf-8")
    return {
        "latest_json": str(latest_path),
        "archived_json": str(archive_path),
        "cases_jsonl": str(cases_path),
        "report_md": str(report_path),
    }


def render_phase1_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary", {})
    averages = summary.get("metric_averages", {})
    lines = [
        "# Phase 1 RAG + Generation Evaluation Report",
        "",
        f"- Generated at: `{result.get('generated_at', '')}`",
        f"- Sample count: `{summary.get('sample_count', 0)}`",
        f"- Needs review rate: `{summary.get('needs_review_rate', 0)}`",
        "",
        "## Metric Averages",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in averages.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Risk Distribution", "", "| Risk | Count |", "|---|---:|"])
    for key, value in (summary.get("risk_distribution") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Worst Cases", "", "| ID | Title | Risks | Evidence | Retrieval | Format |", "|---|---|---|---:|---:|---:|"])
    for item in summary.get("worst_cases", []) or []:
        title = str(item.get("title") or "").replace("|", " ")[:60]
        risks = ",".join(item.get("risk_flags", []) or [])
        lines.append(
            f"| `{item.get('id')}` | {title} | `{risks}` | `{item.get('evidence_coverage')}` | "
            f"`{item.get('retrieval_precision_at_k')}` | `{item.get('format_compliance')}` |"
        )
    return "\n".join(lines) + "\n"
