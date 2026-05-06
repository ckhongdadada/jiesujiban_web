"""Batch quality evaluation for /api/analyze responses."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.jsjb.reply_generation.error_analysis import ReplyErrorExtractor


@dataclass
class BatchQualityEvaluator:
    """Evaluate many messages through a running API service."""

    api_url: str = "http://127.0.0.1:5000/api/analyze"
    timeout_seconds: int = 120

    def evaluate_rows(self, rows: Iterable[Dict[str, Any]], limit: Optional[int] = None) -> Dict[str, Any]:
        records = []
        start = time.time()
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            records.append(self.evaluate_one(row, index=index))
        return {
            "summary": build_batch_summary(records, elapsed_seconds=time.time() - start),
            "records": records,
        }

    def evaluate_one(self, row: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        payload = {
            "tag": row.get("tag") or row.get("留言标签") or "",
            "title": row.get("title") or row.get("留言标题") or "",
            "body": row.get("body") or row.get("留言正文") or "",
            "_debug": bool(row.get("_debug", False)),
        }
        started = time.time()
        try:
            response = post_json(self.api_url, payload, timeout_seconds=self.timeout_seconds)
            elapsed = time.time() - started
            expected_unit = row.get("expected_unit") or row.get("官方回复单位") or row.get("unit") or ""
            reference_reply = row.get("reference_reply") or row.get("官方回复正文") or ""
            quality = evaluate_response_quality(response, expected_unit=expected_unit, reference_reply=reference_reply)
            return {
                "index": index,
                "status": "ok",
                "elapsed_seconds": round(elapsed, 4),
                "input": payload,
                "expected_unit": expected_unit,
                "expected_district": row.get("expected_district") or row.get("district") or "",
                **flatten_api_response(response),
                "quality_attribution": quality,
            }
        except Exception as exc:
            return {
                "index": index,
                "status": "error",
                "elapsed_seconds": round(time.time() - started, 4),
                "input": payload,
                "error": str(exc),
            }


def evaluate_response_quality(
    response: Dict[str, Any],
    expected_unit: str = "",
    reference_reply: str = "",
) -> Dict[str, Any]:
    extractor = ReplyErrorExtractor()
    units = response.get("units") or []
    predicted_unit = units[0].get("unit", "") if units else ""
    return extractor.analyze(
        generated_reply=response.get("reply", ""),
        reference_reply=reference_reply or "",
        retrieval_hits=response.get("retrieval") or [],
        location_result=response.get("location") or {},
        expected_unit=expected_unit or predicted_unit,
        feedback_type="batch_eval",
        comments="",
    )


def flatten_api_response(response: Dict[str, Any]) -> Dict[str, Any]:
    location = response.get("location") or {}
    units = response.get("units") or []
    retrieval = response.get("retrieval") or []
    processing_time = response.get("processing_time") or {}
    return {
        "api_status": response.get("status", ""),
        "district": location.get("district", ""),
        "location_confidence": location.get("confidence", ""),
        "top1_unit": units[0].get("unit", "") if len(units) >= 1 else "",
        "top1_confidence": units[0].get("confidence", "") if len(units) >= 1 else "",
        "top3_units": [item.get("unit", "") for item in units[:3]],
        "retrieval_count": len(retrieval),
        "retrieval_top1_title": retrieval[0].get("title", "") if retrieval else "",
        "retrieval_top1_score": retrieval[0].get("score", "") if retrieval else "",
        "evidence_strength": response.get("evidence_strength", ""),
        "reply_mode": response.get("reply_mode", ""),
        "needs_review": bool(response.get("needs_review")),
        "reply": response.get("reply", ""),
        "processing_total": processing_time.get("total", ""),
        "processing_location": processing_time.get("location", ""),
        "processing_classification": processing_time.get("classification", ""),
        "processing_retrieval": processing_time.get("retrieval", ""),
        "processing_generation": processing_time.get("generation", ""),
    }


def build_batch_summary(records: List[Dict[str, Any]], elapsed_seconds: float = 0.0) -> Dict[str, Any]:
    ok_records = [item for item in records if item.get("status") == "ok"]
    error_records = [item for item in records if item.get("status") != "ok"]
    needs_review = [item for item in ok_records if item.get("needs_review")]
    attribution_counts: Dict[str, int] = {}
    for item in ok_records:
        for error_type in item.get("quality_attribution", {}).get("error_types", []) or []:
            attribution_counts[error_type] = attribution_counts.get(error_type, 0) + 1

    avg_elapsed = (
        sum(float(item.get("elapsed_seconds") or 0.0) for item in ok_records) / len(ok_records)
        if ok_records
        else 0.0
    )
    return {
        "sample_count": len(records),
        "ok_count": len(ok_records),
        "error_count": len(error_records),
        "needs_review_count": len(needs_review),
        "needs_review_rate": round(len(needs_review) / max(len(ok_records), 1), 4),
        "avg_elapsed_seconds": round(avg_elapsed, 4),
        "total_elapsed_seconds": round(elapsed_seconds, 4),
        "quality_attribution_distribution": dict(sorted(attribution_counts.items(), key=lambda item: item[1], reverse=True)),
    }


def post_json(url: str, payload: Dict[str, Any], timeout_seconds: int = 120) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {message}") from exc


def read_input_rows(path: str) -> List[Dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        return json.loads(input_path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(input_path).fillna("").to_dict(orient="records")
    if suffix in {".xlsx", ".xls"}:
        import pandas as pd

        return pd.read_excel(input_path).fillna("").to_dict(orient="records")
    raise ValueError(f"Unsupported input file type: {input_path.suffix}")


def write_outputs(result: Dict[str, Any], output_path: str) -> Dict[str, str]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    written = {"json": str(output)}

    records = result.get("records") or []
    if records:
        try:
            import pandas as pd

            flat_records = []
            for item in records:
                flat = {k: v for k, v in item.items() if k not in {"input", "quality_attribution"}}
                flat["input_title"] = item.get("input", {}).get("title", "")
                flat["input_body"] = item.get("input", {}).get("body", "")
                flat["quality_error_types"] = ",".join(item.get("quality_attribution", {}).get("error_types", []) or [])
                flat["quality_severity"] = item.get("quality_attribution", {}).get("severity", "")
                flat["quality_summary"] = item.get("quality_attribution", {}).get("summary", "")
                flat_records.append(flat)
            csv_path = output.with_suffix(".csv")
            pd.DataFrame(flat_records).to_csv(csv_path, index=False, encoding="utf-8-sig")
            written["csv"] = str(csv_path)
            xlsx_path = output.with_suffix(".xlsx")
            pd.DataFrame(flat_records).to_excel(xlsx_path, index=False)
            written["xlsx"] = str(xlsx_path)
        except Exception as exc:
            written["tabular_error"] = str(exc)
    return written
